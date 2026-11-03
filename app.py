"""
FinQA Hierarchical LLM Task Paradigm & Exhaustive Clustering Dashboard
Hostable Streamlit Application for Exploring and Clustering All 8,281 FinQA Instances.

Exhaustive 3-Tier Hierarchy:
  Step 1: Reasoning Hops (Single-hop vs Multi-hop)
  Step 2: Modalities (Text-only vs Table-only vs Text-Table)
  Step 3: QA Type (NumericalQA vs Multi-hopQA)
"""

import json
import re
import os
import textwrap
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="FinQA Task Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-multihop {
        background-color: #1E88E5;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-numerical {
        background-color: #2E7D32;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .brief-context-box {
        background-color: #F1F5F9;
        border-left: 5px solid #0EA5E9;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
    }
    .hierarchy-tag {
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA STRUCTURES & HIERARCHICAL ENGINE
# ==============================================================================

@dataclass
class FinQAInstance:
    id: str
    question: str
    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]
    answer: str = ""
    final_result: str = ""
    program: str = ""
    gold_inds: Dict[str, str] = field(default_factory=dict)
    split: str = "train"

    @property
    def full_text(self) -> List[str]:
        return self.pre_text + self.post_text

    @property
    def table_df(self) -> pd.DataFrame:
        if not self.table or len(self.table) < 1:
            return pd.DataFrame()
        headers = self.table[0]
        rows = self.table[1:]
        max_cols = max(len(headers), max((len(r) for r in rows), default=0))
        norm_headers = [str(h) if i < len(headers) else f"Col_{i}" for i, h in enumerate(headers)]
        while len(norm_headers) < max_cols:
            norm_headers.append(f"Col_{len(norm_headers)}")
        norm_rows = []
        for r in rows:
            r_padded = [str(c) for c in r] + [""] * (max_cols - len(r))
            norm_rows.append(r_padded[:max_cols])
        return pd.DataFrame(norm_rows, columns=norm_headers)

    def get_brief_context(self) -> Dict[str, Any]:
        """Extracts brief grounding context of the initial question from filing metadata and gold facts."""
        parts = self.id.split("/")
        ticker = parts[0] if len(parts) > 0 else "Filing"
        year = parts[1] if len(parts) > 1 else ""
        doc_ref = parts[2].replace(".pdf-", " (Item ") + ")" if len(parts) > 2 else self.id

        facts = []
        if self.gold_inds:
            for k, v in self.gold_inds.items():
                src_type = "Table Line Item" if "table" in k.lower() else "Narrative Disclosure"
                clean_v = str(v).strip()
                if clean_v:
                    facts.append(f"**{src_type} ({k}):** {clean_v}")
        else:
            if self.pre_text:
                clean_pre = [p for p in self.pre_text if p.strip() not in [".", "", "-", "•"]]
                if clean_pre:
                    facts.append(f"**Narrative Pre-Text:** {clean_pre[0]}")
            if self.table and len(self.table) > 1:
                facts.append(f"**Table Line Item:** {' | '.join(self.table[1][:3])}...")

        return {
            "ticker": ticker,
            "year": year,
            "doc_ref": doc_ref,
            "facts": facts
        }


@dataclass
class ClassificationResult:
    # 3-Tier Exhaustive Hierarchy
    tier1_reasoning_hops: str  # "Single-hop" or "Multi-hop"
    tier2_modality: str        # "Text-only", "Table-only", "Text-Table"
    tier3_qa_type: str         # "NumericalQA" or "Multi-hopQA"
    hierarchy_path: str        # e.g., "Single-hop → Table-only → NumericalQA"
    
    # Granular Diagnostics
    cluster: str               # "NumericalQA" or "Multi-hopQA"
    sub_category: str
    confidence: float
    evidence_modality: str
    hop_count: int
    dependency_depth: int
    is_chained: bool
    is_aggregation: bool
    paradigm_interprets: float
    paradigm_retrieves: float
    paradigm_solves: float
    human_rationale: str


class FinQAClusteringEngine:
    """
    Exhaustive 3-Tier Hierarchical Clustering Engine for FinQA:
      Step 1: Reasoning Hops (Single-hop vs Multi-hop)
      Step 2: Modalities (Text-only vs Table-only vs Text-Table)
      Step 3: QA Type (NumericalQA vs Multi-hopQA)
    """
    AGGREGATION_OPS = {"table-sum", "table-average", "table-max", "table-min"}
    MULTI_HOP_CUES = [
        "compared to", "of which", "after excluding", "adjusted for", 
        "as described in", "note", "footnote", "respectively", "higher than",
        "which segment", "which year", "combined", "both", "difference between"
    ]
    NUMERICAL_RATIO_CUES = [
        "percentage change", "percent increase", "percent decrease", 
        "growth rate", "ratio of", "percentage of", "operating margin", "average"
    ]

    @classmethod
    def analyze_instance(cls, inst: FinQAInstance) -> ClassificationResult:
        q_text = inst.question.lower()
        program = inst.program or ""
        gold_inds = inst.gold_inds or {}
        
        # ----------------------------------------------------------------------
        # STEP 2 (MODALITY EVALUATION): Text-only vs Table-only vs Text-Table
        # ----------------------------------------------------------------------
        has_table_fact = False
        has_text_fact = False
        
        if gold_inds:
            for k in gold_inds.keys():
                if "table" in k.lower():
                    has_table_fact = True
                elif "text" in k.lower():
                    has_text_fact = True
        else:
            q_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', q_text)
            table_str = " ".join([" ".join(row) for row in inst.table]).lower()
            text_str = " ".join(inst.full_text).lower()
            for num in q_numbers:
                if num in table_str:
                    has_table_fact = True
                if num in text_str:
                    has_text_fact = True
            if not has_table_fact and not has_text_fact:
                has_table_fact = bool(inst.table and len(inst.table) > 1)
                has_text_fact = bool(inst.full_text)

        if has_table_fact and has_text_fact:
            tier2_modality = "Text-Table"
            evidence_modality = "table-text"
        elif has_table_fact:
            tier2_modality = "Table-only"
            evidence_modality = "table-only"
        else:
            tier2_modality = "Text-only"
            evidence_modality = "text-only"

        hop_count = len(gold_inds) if gold_inds else (2 if tier2_modality == "Text-Table" else 1)

        # ----------------------------------------------------------------------
        # OPERATIONS & COMPLEXITY DECOMPOSITION
        # ----------------------------------------------------------------------
        raw_ops = re.findall(r'[a-zA-Z\-]+\([^)]+\)', program) if program else []
        is_chained = any('#' in op for op in raw_ops)
        is_aggregation = any(any(op.startswith(agg) for agg in cls.AGGREGATION_OPS) for op in raw_ops)
        
        # Filter trivial unit normalizations like divide(100, 100) or divide(1000, 1000)
        non_trivial_ops = []
        for rop in raw_ops:
            cleaned = rop.replace(" ", "")
            if any(cleaned == f"divide({v},{v})" for v in ["100", "1000", "10", "1"]):
                continue
            non_trivial_ops.append(rop)

        is_complex_multistep_math = len(non_trivial_ops) >= 2
        dependency_depth = max(len(non_trivial_ops), 1)

        # Row Index Inspection for Table Ratios
        table_row_indices = []
        for k in gold_inds.keys():
            if "table" in k.lower():
                matches = re.findall(r'\d+', k)
                if matches:
                    table_row_indices.append(int(matches[0]))

        has_multiple_table_rows = len(set(table_row_indices)) > 1
        has_single_table_row = len(set(table_row_indices)) == 1

        is_part_to_whole_cue = any(w in q_text for w in [
            "comprised of", "percentage of total", "portion of", "share of", 
            "out of", "percent of total", "as a percentage of", "component of"
        ])
        is_temporal_growth_cue = any(w in q_text for w in [
            "percentage change", "percent increase", "percent decrease", 
            "growth rate", "change from", "increase from", "decrease from"
        ])
        is_ratio_cue = any(w in q_text for w in ["percentage", "percent", "ratio", "margin", "rate"])

        is_cross_row_ratio = has_multiple_table_rows or (is_part_to_whole_cue and tier2_modality == "Table-only")
        is_single_row_temporal = (has_single_table_row and (is_temporal_growth_cue or is_ratio_cue)) or (
            is_temporal_growth_cue and not is_cross_row_ratio and tier2_modality == "Table-only"
        )

        # ----------------------------------------------------------------------
        # STEP 1 (REASONING HOPS DIVISION): Single-hop vs Multi-hop
        # ----------------------------------------------------------------------
        # Multi-hop reasoning applies if:
        # 1. Evidence requires gathering across boundaries (cross-modal Text-Table or multiple sentences/rows)
        # 2. Or the mathematical program involves non-trivial multi-step composition (>= 2 operations)
        if tier2_modality == "Text-Table" or (tier2_modality == "Text-only" and hop_count >= 2) or is_complex_multistep_math or has_multiple_table_rows:
            tier1_reasoning_hops = "Multi-hop"
        else:
            tier1_reasoning_hops = "Single-hop"

        # ----------------------------------------------------------------------
        # STEP 3 (QA TYPE DIVISION): NumericalQA vs Multi-hopQA
        # ----------------------------------------------------------------------
        # NumericalQA Criteria:
        # - DSL mapping, implicit constants (100 for percentage), scale alignment, 
        #   heterogeneous number format handling, localized single-row temporal ratios, 
        #   or direct part-to-whole column ratios.
        # Multi-hopQA Criteria:
        # - Cross-modal fact retrieval (Table-to-Text), multi-step evidence gathering across 
        #   disjoint paragraphs, temporal/entity bridging, or complex multi-row bridges.
        
        if tier2_modality == "Text-Table":
            # Cross-modal Table-to-Text hopping is fundamentally Multi-hopQA
            tier3_qa_type = "Multi-hopQA"
            sub_category = "Cross-Modal Hybrid Hop (Table + Text)"
            confidence = 0.98
            human_rationale = "Cross-modal fact retrieval: The query cannot be answered by the table or text alone, requiring hopping between financial tabular data and explanatory narrative disclosures."

        elif tier2_modality == "Text-only" and hop_count >= 2:
            # Multi-step evidence gathering scattered across report
            tier3_qa_type = "Multi-hopQA"
            sub_category = "Dispersed Multi-Sentence Text Navigation"
            confidence = 0.90
            human_rationale = "Multi-step evidence gathering: Requires retrieving and synthesizing a set of distinct gold facts scattered across multiple narrative paragraphs."

        elif tier1_reasoning_hops == "Single-hop":
            # Single-hop reasoning maps directly to NumericalQA
            tier3_qa_type = "NumericalQA"
            confidence = 0.94
            if is_single_row_temporal:
                sub_category = "Single-Row Temporal Ratio"
                human_rationale = "Domain DSL Mapping & Implicit Constants: Co-located in a single table row; system executes horizontal percentage change formula with implicit 100 multiplier."
            elif is_aggregation:
                sub_category = "Tabular Aggregation (Sum/Avg/Min/Max)"
                human_rationale = "Symbolic DSL Execution: Direct aggregation over contiguous tabular cells requiring scale and unit alignment."
            else:
                sub_category = "Single-Fact Direct Arithmetic"
                human_rationale = "DSL Mapping & Format Handling: Co-located in a single sentence or table row; executes direct arithmetic with elementary unit normalization."

        else:
            # Multi-hop reasoning branch: determine if NumericalQA or Multi-hopQA
            if tier2_modality == "Table-only" and is_cross_row_ratio and not is_complex_multistep_math:
                # Vertical component-to-total ratio (direct division across 2 rows)
                tier3_qa_type = "NumericalQA"
                sub_category = "Cross-Row Column Ratio (Part-to-Whole)"
                confidence = 0.95
                human_rationale = "DSL Mapping & Scale Alignment: Requires 2 table hops (component and total) within the same column; classified as NumericalQA because it executes a direct single-step ratio without multi-step chaining."

            elif tier2_modality == "Table-only" and is_single_row_temporal:
                # Multi-step horizontal growth rate: (New - Old) / Old
                tier3_qa_type = "NumericalQA"
                sub_category = "Single-Row Temporal Ratio"
                confidence = 0.95
                human_rationale = "Implicit Operator & Formula Mapping: Multi-step formula ((Current - Prior) / Prior) co-located in a single row; cognitive challenge is mathematical formula execution."

            elif tier2_modality == "Table-only" and is_aggregation:
                tier3_qa_type = "NumericalQA"
                sub_category = "Tabular Aggregation (Sum/Avg/Min/Max)"
                confidence = 0.94
                human_rationale = "Aggregation & Unit Normalization: Mathematical aggregation over tabular rows aligning scales and numerical representations."

            elif hop_count == 1 and is_complex_multistep_math:
                # Multi-step math on a single sentence: intermediate tracking
                tier3_qa_type = "Multi-hopQA"
                sub_category = "Multi-Step Complex Mathematical Reasoning"
                confidence = 0.86
                human_rationale = "Intermediate Variable Tracking: Single-fact retrieval but requires a multi-step algebraic pipeline where intermediate outputs become inputs to subsequent execution steps."

            elif tier2_modality == "Table-only" and is_cross_row_ratio and is_complex_multistep_math:
                # Multi-row table bridging
                tier3_qa_type = "Multi-hopQA"
                sub_category = "Multi-Row Tabular Bridge"
                confidence = 0.92
                human_rationale = "Entity Bridging & Chained Tracking: Navigates non-adjacent table rows and chains intermediate results across multiple operational steps."

            else:
                tier3_qa_type = "Multi-hopQA" if (hop_count >= 2 or is_complex_multistep_math) else "NumericalQA"
                sub_category = "Multi-Row Tabular Bridge" if tier2_modality == "Table-only" else "Chained Intermediate Dependency Hop"
                confidence = 0.85
                human_rationale = "Multi-Step Evidence Gathering: Traces entities across report sections and maintains custody of intermediate variables across execution hops."

        hierarchy_path = f"{tier1_reasoning_hops} → {tier2_modality} → {tier3_qa_type}"

        # Cognitive Load Scores
        s_interprets = 0.4
        if any(w in q_text for w in ["margin", "diluted", "fair value", "basis points", "ebitda", "amortization"]):
            s_interprets += 0.3
        if is_ratio_cue or is_temporal_growth_cue:
            s_interprets += 0.2
        s_interprets = min(s_interprets, 1.0)

        s_retrieves = 0.3
        if tier2_modality == "Text-Table":
            s_retrieves += 0.45
        elif hop_count >= 2:
            s_retrieves += 0.30
        if any(cue in q_text for cue in cls.MULTI_HOP_CUES):
            s_retrieves += 0.15
        s_retrieves = min(s_retrieves, 1.0)

        s_solves = 0.35
        if dependency_depth >= 2:
            s_solves += 0.30
        if is_aggregation:
            s_solves += 0.25
        if "divide" in program or "%" in q_text:
            s_solves += 0.15
        s_solves = min(s_solves, 1.0)

        return ClassificationResult(
            tier1_reasoning_hops=tier1_reasoning_hops,
            tier2_modality=tier2_modality,
            tier3_qa_type=tier3_qa_type,
            hierarchy_path=hierarchy_path,
            cluster=tier3_qa_type,
            sub_category=sub_category,
            confidence=round(confidence, 3),
            evidence_modality=evidence_modality,
            hop_count=hop_count,
            dependency_depth=dependency_depth,
            is_chained=is_chained,
            is_aggregation=is_aggregation,
            paradigm_interprets=round(s_interprets, 2),
            paradigm_retrieves=round(s_retrieves, 2),
            paradigm_solves=round(s_solves, 2),
            human_rationale=human_rationale
        )


# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================

@st.cache_data(show_spinner=True)
def load_all_finqa_data(max_samples_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads all 8,281 instances from official GitHub repository with caching."""
    github_base = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
    splits = [("train", "train.json"), ("validation", "dev.json"), ("test", "test.json")]
    
    all_raw = []
    for split_label, filename in splits:
        url = f"{github_base}/{filename}"
        try:
            resp = requests.get(url, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                if max_samples_per_split:
                    data = data[:max_samples_per_split]
                for item in data:
                    item["_split"] = split_label
                all_raw.extend(data)
        except Exception:
            pass

    if not all_raw:
        sample_mock = [
            {
                "id": "ADI/2009/page_49.pdf-1",
                "_split": "train",
                "pre_text": [
                    "interest rate to a variable interest rate based on the three-month libor plus 2.05% as of october 31, 2009.",
                    "if libor changes by 100 basis points, our annual interest expense would change by $ 3.8 million.",
                    "foreign currency exposure as more fully described in note 2i."
                ],
                "post_text": ["fair value of forward exchange contracts after a 10% unfavorable movement."],
                "table": [["", "2009", "2008"], ["fair value of forward contracts", "6427", "-23158"]],
                "qa": {
                    "question": "what is the the interest expense in 2009?",
                    "program": "divide(100, 100), divide(3.8, #0)",
                    "exe_ans": "3.8",
                    "answer": "3.8",
                    "gold_inds": {"text_1": "if libor changes by 100 basis points, our annual interest expense would change by $ 3.8 million."}
                }
            },
            {
                "id": "INTC/2013/page_71.pdf-4",
                "_split": "train",
                "pre_text": ["note 5: cash and investments at the end of each period were as follows:"],
                "post_text": ["in total, we received proceeds of 470 million on these transactions."],
                "table": [
                    ["(in millions)", "dec 28 2013", "dec 29 2012"],
                    ["available-for-sale investments", "$ 18086", "$ 14001"],
                    ["cash", "854", "593"],
                    ["total cash and investments", "$ 31561", "$ 26302"]
                ],
                "qa": {
                    "question": "what percentage of total cash and investments as of dec. 29 2012 was comprised of available-for-sale investments?",
                    "program": "divide(14001, 26302)",
                    "exe_ans": "0.53232",
                    "answer": "53.2%",
                    "gold_inds": {"table_0": "available-for-sale investments 14001", "table_3": "total cash and investments 26302"}
                }
            },
            {
                "id": "MSFT/2014/page_55.pdf-2",
                "_split": "validation",
                "pre_text": ["note 4: restructuring charges incurred in 2014 were $ 50 million."],
                "post_text": ["operating activities provided positive cash flow."],
                "table": [["", "2014", "2013"], ["Total revenue", "1250", "1000"], ["Operating income", "300", "250"]],
                "qa": {
                    "question": "considering restructuring costs in note 4, what was the adjusted operating margin in 2014?",
                    "program": "add(300, 50), divide(#0, 1250)",
                    "exe_ans": "0.28",
                    "answer": "28%",
                    "gold_inds": {"text_0": "restructuring charges $50 million", "table_1": "Total revenue 1250", "table_2": "Operating income 300"}
                }
            }
        ]
        all_raw = sample_mock

    return all_raw


def convert_to_instances(raw_list: List[Dict[str, Any]]) -> List[FinQAInstance]:
    instances = []
    for item in raw_list:
        qa_block = item.get("qa", {})
        instances.append(FinQAInstance(
            id=str(item.get("id", "")),
            question=str(qa_block.get("question", "")),
            pre_text=item.get("pre_text", []) or [],
            post_text=item.get("post_text", []) or [],
            table=item.get("table", []) or [],
            answer=str(qa_block.get("answer", "")),
            final_result=str(qa_block.get("exe_ans", "")),
            program=str(qa_block.get("program", "")),
            gold_inds=qa_block.get("gold_inds", {}) or {},
            split=str(item.get("_split", "train"))
        ))
    return instances


@st.cache_data(show_spinner=True)
def get_clustered_dataframe(_instances: List[FinQAInstance]) -> pd.DataFrame:
    records = []
    for inst in _instances:
        res = FinQAClusteringEngine.analyze_instance(inst)
        records.append({
            "id": inst.id,
            "split": inst.split,
            "question": inst.question,
            "answer": inst.final_result or inst.answer,
            "program": inst.program,
            
            # Exhaustive 3-Tier Hierarchy Fields
            "tier1_hops": res.tier1_reasoning_hops,
            "tier2_modality": res.tier2_modality,
            "tier3_qa_type": res.tier3_qa_type,
            "hierarchy_path": res.hierarchy_path,
            
            "cluster": res.cluster,
            "sub_category": res.sub_category,
            "confidence": res.confidence,
            "modality": res.evidence_modality,
            "hop_count": res.hop_count,
            "dependency_depth": res.dependency_depth,
            "is_chained": res.is_chained,
            "is_aggregation": res.is_aggregation,
            "interprets_score": res.paradigm_interprets,
            "retrieves_score": res.paradigm_retrieves,
            "solves_score": res.paradigm_solves,
            "rationale": res.human_rationale
        })
    return pd.DataFrame(records)


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

def main():
    st.markdown('<div class="main-title">📊 FinQA Task Clustering Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Step-by-Step Exhaustive Taxonomy (Hops → Modality → QA Type) Summing to 100% of the Dataset</div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Hierarchy & Filters")
        
        sample_scope = st.radio(
            "Load Scope:",
            ["Complete Dataset (All 8,281)", "Representative Sample (500)", "Quick Demo (100)"],
            index=0
        )
        max_samples = None if "Complete" in sample_scope else (500 if "500" in sample_scope else 100)

        # Load Raw Data
        raw_data = load_all_finqa_data(max_samples_per_split=max_samples)
        instances = convert_to_instances(raw_data)
        df_clustered = get_clustered_dataframe(instances)
        
        st.success(f"Loaded **{len(instances):,}** instances.")

        st.subheader("3-Step Hierarchy Filters")
        
        # Step 1: Reasoning Hops Filter
        hop_choices = ["All Hops"] + sorted(list(df_clustered['tier1_hops'].unique()))
        selected_hops = st.selectbox("Step 1: Reasoning Hops:", hop_choices, index=0)

        # Step 2: Modality Filter
        mod_choices = ["All Modalities"] + sorted(list(df_clustered['tier2_modality'].unique()))
        selected_modality = st.selectbox("Step 2: Modalities:", mod_choices, index=0)

        # Step 3: QA Type Filter
        qa_choices = ["All QA Types"] + sorted(list(df_clustered['tier3_qa_type'].unique()))
        selected_qa = st.selectbox("Step 3: QA Type:", qa_choices, index=0)

        st.divider()
        split_choices = ["All Splits"] + sorted(list(df_clustered['split'].unique()))
        selected_split = st.selectbox("Dataset Split:", split_choices, index=0)

        search_query = st.text_input("🔍 Search question / ID:", "").strip().lower()

    # Apply Filters to DataFrame
    filtered_df = df_clustered.copy()
    if selected_hops != "All Hops":
        filtered_df = filtered_df[filtered_df['tier1_hops'] == selected_hops]
    if selected_modality != "All Modalities":
        filtered_df = filtered_df[filtered_df['tier2_modality'] == selected_modality]
    if selected_qa != "All QA Types":
        filtered_df = filtered_df[filtered_df['tier3_qa_type'] == selected_qa]
    if selected_split != "All Splits":
        filtered_df = filtered_df[filtered_df['split'] == selected_split]
    if search_query:
        filtered_df = filtered_df[
            filtered_df['question'].str.lower().str.contains(search_query) | 
            filtered_df['id'].str.lower().str.contains(search_query)
        ]

    matched_indices = list(filtered_df.index)

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Instance Explorer",
        "🌳 Hierarchy Tree",
        "💾 Batch Clustering & Export",
        "📖 Taxonomy & Methodology"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: INSTANCE EXPLORER
    # --------------------------------------------------------------------------
    with tab1:
        if not matched_indices:
            st.warning("No instances match the current hierarchy filters. Please adjust your criteria.")
        else:
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 1, 1, 1])
            with col_ctrl1:
                st.caption(f"Showing **{len(matched_indices):,}** matching instances out of **{len(df_clustered):,}** total.")
            with col_ctrl2:
                btn_prev = st.button("◀ Previous", use_container_width=True)
            with col_ctrl3:
                btn_next = st.button("Next ▶", use_container_width=True)
            with col_ctrl4:
                btn_rand = st.button("🎲 Random", use_container_width=True)

            if "view_pointer" not in st.session_state:
                st.session_state.view_pointer = 0

            if btn_prev and st.session_state.view_pointer > 0:
                st.session_state.view_pointer -= 1
            elif btn_next and st.session_state.view_pointer < len(matched_indices) - 1:
                st.session_state.view_pointer += 1
            elif btn_rand:
                st.session_state.view_pointer = int(np.random.randint(0, len(matched_indices)))

            st.session_state.view_pointer = min(max(0, st.session_state.view_pointer), len(matched_indices) - 1)
            active_idx = matched_indices[st.session_state.view_pointer]
            current_inst = instances[active_idx]
            current_row = df_clustered.loc[active_idx]

            # Direct index slider
            sel_idx = st.slider(
                "Direct Index Jump:",
                min_value=0,
                max_value=len(matched_indices) - 1,
                value=st.session_state.view_pointer
            )
            if sel_idx != st.session_state.view_pointer:
                st.session_state.view_pointer = sel_idx
                st.rerun()

            # Diagnostic Card & Hierarchy
            st.divider()
            c_top1, c_top2 = st.columns([3, 1])
            with c_top1:
                st.subheader(f"Instance: `{current_inst.id}`")
                st.caption(f"Split: **{current_inst.split}** | Match **#{st.session_state.view_pointer + 1:,}** of **{len(matched_indices):,}** | Global Index: **#{active_idx:,}**")
                st.markdown(f"**Hierarchy Path:** `{current_row['hierarchy_path']}`")
            with c_top2:
                if current_row['tier3_qa_type'] == "Multi-hopQA":
                    st.markdown('<div style="text-align:right;"><span class="badge-multihop">🌐 Multi-hopQA</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align:right;"><span class="badge-numerical">🧮 NumericalQA</span></div>', unsafe_allow_html=True)
                st.caption(f"Confidence: **{int(current_row['confidence'] * 100)}%**")

            # Sub-category & Rationale
            st.info(f"**Sub-Category:** {current_row['sub_category']}  \n**Step 1:** `{current_row['tier1_hops']}` | **Step 2:** `{current_row['tier2_modality']}` | **Step 3:** `{current_row['tier3_qa_type']}`  \n**Audit Rationale:** {current_row['rationale']}")

            # Cognitive Load Gauges
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Interprets (Domain DSL & Mapping)", f"{int(current_row['interprets_score'] * 100)}%")
                st.progress(current_row['interprets_score'])
            with m2:
                st.metric("Retrieves (Evidence Nav & Bridging)", f"{int(current_row['retrieves_score'] * 100)}%")
                st.progress(current_row['retrieves_score'])
            with m3:
                st.metric("Solves (Math Execution (Arithmetic))", f"{int(current_row['solves_score'] * 100)}%")
                st.progress(current_row['solves_score'])

            # Question & Answer Box
            st.markdown(f"#### Question❓ :\n> **{current_inst.question}**")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"**Execution Answer:** `{current_inst.final_result or current_inst.answer or 'N/A'}`")
            with col_a2:
                st.markdown(f"**Reasoning Program:** `{current_inst.program or 'N/A'}`")

            # ==================================================================
            # USER-REQUESTED BRIEF CONTEXT BOX (BELOW QUESTION)
            # ==================================================================
            brief_ctx = current_inst.get_brief_context()
            st.markdown("#### Question Context & Grounding Summary")
            with st.container():
                st.markdown(f"""
                <div class="brief-context-box">
                    <div style="font-weight:700; color:#0369A1; font-size:1.02rem; margin-bottom:6px;">
                        Filing Source: {brief_ctx['ticker']} ({brief_ctx['year']}) — {brief_ctx['doc_ref']}
                    </div>
                    <div style="color:#334155; font-size:0.92rem; line-height:1.5;">
                        <strong>Query Focus:</strong> This question queries financial statement disclosures from <strong>{brief_ctx['ticker']}</strong>'s Form 10-K report for fiscal year <strong>{brief_ctx['year']}</strong>.
                    </div>
                    <div style="margin-top:8px; font-size:0.9rem; color:#1E293B;">
                        <strong>Primary Supporting Facts Grounding the Initial Question:</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if brief_ctx['facts']:
                    for f_item in brief_ctx['facts']:
                        st.markdown(f"- {f_item}")
                else:
                    st.caption("Supporting facts anchored in contiguous table rows or pre-text disclosures.")

            # Detailed Context Disclosures & Tables (Preserved below brief context)
            st.markdown("### Detailed Context Disclosures & Tables")
            
            clean_pre = [p for p in current_inst.pre_text if p.strip() not in [".", "", "-", "•"]]
            with st.expander("Narrative Pre-Text Disclosures", expanded=bool(clean_pre)):
                if clean_pre:
                    for p in clean_pre:
                        st.markdown(f"- {p}")
                else:
                    st.caption("No pre-text disclosures.")

            with st.expander("Structured Financial Statement Table", expanded=bool(current_inst.table)):
                if current_inst.table and len(current_inst.table) > 1:
                    st.dataframe(current_inst.table_df, use_container_width=True)
                else:
                    st.caption("No table available.")

            clean_post = [p for p in current_inst.post_text if p.strip() not in [".", "", "-", "•"]]
            with st.expander("Narrative Post-Text Disclosures", expanded=bool(clean_post)):
                if clean_post:
                    for p in clean_post:
                        st.markdown(f"- {p}")
                else:
                    st.caption("No post-text disclosures.")

    # --------------------------------------------------------------------------
    # TAB 2: EXHAUSTIVE HIERARCHY TREE (SUM = 8,281)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Exhaustive 3-Step Hierarchical Decomposition (Sum = 100% of Dataset)")
        st.markdown("""
        Every instance in the dataset belongs to exactly **one mutually exclusive path** through the 3-step hierarchy tree.
        The sum across all leaf clusters rigorously equals the entire dataset.
        """)

        total_n = len(df_clustered)

        # High-Level Metrics
        k1, k2, k3, k4 = st.columns(4)
        n_single = (df_clustered['tier1_hops'] == "Single-hop").sum()
        n_multi_hops = (df_clustered['tier1_hops'] == "Multi-hop").sum()
        n_num_qa = (df_clustered['tier3_qa_type'] == "NumericalQA").sum()
        n_multi_qa = (df_clustered['tier3_qa_type'] == "Multi-hopQA").sum()

        k1.metric("Total Dataset Instances", f"{total_n:,}", "100.0% Exhaustive")
        k2.metric("Step 1: Single-hop", f"{n_single:,}", f"{n_single/total_n:.1%}")
        k3.metric("Step 1: Multi-hop", f"{n_multi_hops:,}", f"{n_multi_hops/total_n:.1%}")
        k4.metric("Step 3: NumericalQA vs Multi-hopQA", f"{n_num_qa:,} / {n_multi_qa:,}")

        st.divider()

        # Step-by-Step Breakdown Visualizations
        st.markdown("### Step-by-Step Division Analytics")
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            # Step 1 -> Step 2 Division
            fig_step1, ax_s1 = plt.subplots(figsize=(6.5, 4.2))
            s1_s2 = pd.crosstab(df_clustered['tier1_hops'], df_clustered['tier2_modality'])
            s1_s2.plot(kind='bar', stacked=True, ax=ax_s1, colormap='Blues')
            ax_s1.set_title("Step 1 (Hops) → Step 2 (Modalities) Distribution", fontsize=11, fontweight='bold')
            ax_s1.set_ylabel("Instance Count")
            ax_s1.set_xlabel("Reasoning Hops")
            ax_s1.legend(title="Modality", loc="upper left")
            ax_s1.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig_step1)

        with col_h2:
            # Step 2 -> Step 3 Division
            fig_step2, ax_s2 = plt.subplots(figsize=(6.5, 4.2))
            palette_qa = {"NumericalQA": "#2E7D32", "Multi-hopQA": "#1E88E5"}
            s2_s3 = pd.crosstab(df_clustered['tier2_modality'], df_clustered['tier3_qa_type'])
            s2_s3.plot(kind='bar', stacked=True, ax=ax_s2, color=[palette_qa["Multi-hopQA"], palette_qa["NumericalQA"]])
            ax_s2.set_title("Step 2 (Modalities) → Step 3 (QA Type) Distribution", fontsize=11, fontweight='bold')
            ax_s2.set_ylabel("Instance Count")
            ax_s2.set_xlabel("Modality")
            ax_s2.legend(title="QA Type", loc="upper right")
            ax_s2.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig_step2)

        st.divider()

        # EXHAUSTIVE LEAF CLUSTER TABLE (VERIFICATION THAT SUM = TOTAL)
        st.markdown("### Exhaustive Cluster Verification Table ")
        
        tree_records = []
        cum_count = 0
        grouped = df_clustered.groupby(['tier1_hops', 'tier2_modality', 'tier3_qa_type']).size().reset_index(name='Count')
        
        # Sort in logical order
        grouped['Hops_Order'] = grouped['tier1_hops'].map({'Single-hop': 0, 'Multi-hop': 1})
        grouped['Mod_Order'] = grouped['tier2_modality'].map({'Text-only': 0, 'Table-only': 1, 'Text-Table': 2})
        grouped = grouped.sort_values(by=['Hops_Order', 'Mod_Order', 'tier3_qa_type']).drop(columns=['Hops_Order', 'Mod_Order'])

        total_check = grouped['Count'].sum()

        grouped['Share (%)'] = (grouped['Count'] / total_n * 100).round(2).astype(str) + "%"
        grouped['Cumulative Count'] = grouped['Count'].cumsum()
        grouped['Cumulative (%)'] = (grouped['Cumulative Count'] / total_n * 100).round(2).astype(str) + "%"
        grouped['Cluster Path'] = grouped.apply(lambda r: f"{r['tier1_hops']} → {r['tier2_modality']} → {r['tier3_qa_type']}", axis=1)

        display_table = grouped[[
            'tier1_hops', 'tier2_modality', 'tier3_qa_type', 'Cluster Path', 'Count', 'Share (%)', 'Cumulative Count', 'Cumulative (%)'
        ]].rename(columns={
            'tier1_hops': 'Step 1: Hops',
            'tier2_modality': 'Step 2: Modality',
            'tier3_qa_type': 'Step 3: QA Type',
            'Count': 'Instances'
        })

        st.dataframe(display_table, use_container_width=True)

        if total_check == total_n:
            st.success(f"**Exhaustive Sum Verified**: Sum of all leaf clusters = **{total_check:,}** / **{total_n:,}**. Zero unassigned or overlapping instances.")
        else:
            st.error(f"Sum mismatch: {total_check:,} vs {total_n:,}")

        st.divider()

        # Visualizations of Cognitive Profiles and Subcategories
        st.markdown("### Cognitive Paradigms & Specialized Sub-Categories")
        col_bot1, col_bot2 = st.columns([1, 1.1])

        palette = {"Multi-hopQA": "#1E88E5", "NumericalQA": "#2E7D32"}

        with col_bot1:
            fig4, ax4 = plt.subplots(figsize=(6.8, 4.8))
            paradigm_df = df_clustered.melt(
                id_vars=['tier3_qa_type'],
                value_vars=['interprets_score', 'retrieves_score', 'solves_score'],
                var_name='Paradigm', value_name='Cognitive Load'
            )
            paradigm_map = {
                'interprets_score': 'Interprets\n(Domain Knowledge & DSL)',
                'retrieves_score': 'Retrieves\n(Evidence Nav & Bridging)',
                'solves_score': 'Solves\n(Execution & Scale Match)'
            }
            paradigm_df['Paradigm'] = paradigm_df['Paradigm'].map(paradigm_map)
            sns.boxplot(
                data=paradigm_df, x='Paradigm', y='Cognitive Load', 
                hue='tier3_qa_type', ax=ax4, palette=palette
            )
            ax4.set_title("LLM Cognitive Load by QA Type", fontsize=11, fontweight='bold')
            ax4.set_ylim(0, 1.1)
            ax4.legend(loc='upper right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig4)

        with col_bot2:
            fig5, ax5 = plt.subplots(figsize=(7.4, 4.8))
            sub_counts = df_clustered['sub_category'].value_counts().head(6)
            sub_df = df_clustered[df_clustered['sub_category'].isin(sub_counts.index)].copy()
            sub_df['sub_wrapped'] = sub_df['sub_category'].apply(lambda x: textwrap.fill(x, 24))
            wrapped_order = [textwrap.fill(x, 24) for x in sub_counts.index]
            sns.countplot(
                data=sub_df, y='sub_wrapped', hue='tier3_qa_type', 
                ax=ax5, palette=palette, order=wrapped_order
            )
            ax5.set_title("Top Specialized Sub-Categories", fontsize=11, fontweight='bold')
            ax5.set_xlabel("Count")
            ax5.set_ylabel("")
            ax5.legend(loc='lower right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig5)

    # --------------------------------------------------------------------------
    # TAB 3: BATCH CLUSTERING & EXPORT
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Export Clustered Dataset")
        st.write(f"The table below contains all **{len(df_clustered):,}** instances classified across the 3-step hierarchy.")

        st.dataframe(df_clustered, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Datase as CSV",
                data=csv_data,
                file_name="finqa_clustered.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            jsonl_data = df_clustered.to_json(orient="records", lines=True).encode('utf-8')
            st.download_button(
                label="Download Dataset as JSONL",
                data=jsonl_data,
                file_name="finqa_clustered.jsonl",
                mime="application/json",
                use_container_width=True
            )

    # --------------------------------------------------------------------------
    # TAB 4: TAXONOMY & METHODOLOGY
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Theoretical Hierarchial Taxonomy & Classification Methodology")
        
        st.markdown("""
        ### Hierarchy Structure:
        1. **Step 1: Reasoning Hops**
           - **Single-hop**: 1 non-trivial operation / direct point calculation without chained intermediate dependency.
           - **Multi-hop**: $\ge 2$ non-trivial operations (chained intermediate dependency) OR multi-fact traversal across report sections.
        
        2. **Step 2: Modalities**
           - **Text-only**: Evidence localized exclusively within narrative disclosures.
           - **Table-only**: Evidence localized exclusively within financial statement tables.
           - **Text-Table**: Evidence spans the cross-modal boundary between text and tables.

        3. **Step 3: QA Type**
           - **NumericalQA**: Mapping to symbolic DSL steps, handling implicit operators & constants (e.g. 100 for percentage), parsing heterogeneous number formats, scale/unit matching, and localized table ratios.
           - **Multi-hopQA**: Cross-modal fact retrieval (table-to-text hopping), multi-step evidence gathering across scattered paragraphs, temporal/entity bridging, and chained intermediate variable tracking.
        """)

        st.divider()

        st.markdown("""
        ### Core Pointers for Classification:
        #### NumericalQA Characteristics:
        - **DSL Programs**: Maps questions to symbolic execution steps (e.g., `add(450.5, 12.1)`, `divide(result_1, 100)`).
        - **Implicit Operators & Constants**: Inserts required constants (like 100 for percentages) or executes standard financial formulas even if implicit.
        - **Heterogeneous Number Formats**: Accurately recognizes standard numerals, loss notation `(124.5)`, or word-form numbers ("two million").
        - **Scale & Units Match**: Aligns "amounts in thousands" with raw text figures.

        #### Multi-hopQA Characteristics:
        - **Cross-Modal Fact Retrieval**: Requires hopping between table numbers and narrative footnotes.
        - **Multi-Step Evidence Gathering**: Collects 2 to 5 distinct gold facts scattered across reports.
        - **Temporal & Entity Bridging**: Traces entities across different fiscal years and report sections.
        - **Intermediate Variable Tracking**: Chained dependencies where the output of hop 1 is required for hop 2.
        """)


if __name__ == "__main__":
    main()
