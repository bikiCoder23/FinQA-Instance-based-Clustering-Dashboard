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
    page_title="FinQA LLM Task Clustering Dashboard",
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
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA STRUCTURES & ENGINE
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


@dataclass
class ClassificationResult:
    cluster: str  # "Multi-hop QA" or "Numerical QA"
    sub_category: str
    confidence: float
    evidence_modality: str  # "table-only", "text-only", "table-text"
    hop_count: int
    dependency_depth: int
    is_chained: bool
    is_aggregation: bool
    paradigm_interprets: float  # 0.0 - 1.0
    paradigm_retrieves: float   # 0.0 - 1.0
    paradigm_solves: float      # 0.0 - 1.0
    human_rationale: str


class FinQAClusteringEngine:
    """Human-calibrated clustering engine modeling LLM reasoning paradigms."""
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
            modality = "table-text"
        elif has_table_fact:
            modality = "table-only"
        else:
            modality = "text-only"

        hop_count = len(gold_inds) if gold_inds else (2 if modality == "table-text" else 1)

        # 2. Operations & Mathematical Complexity Evaluation (Rule 1)
        raw_ops = re.findall(r'[a-zA-Z\-]+\([^)]+\)', program) if program else []
        is_chained = any('#' in op for op in raw_ops)
        is_aggregation = any(any(op.startswith(agg) for agg in cls.AGGREGATION_OPS) for op in raw_ops)
        
        # Filter out trivial identity normalizations like divide(100, 100) or divide(1000, 1000)
        non_trivial_ops = []
        for rop in raw_ops:
            cleaned = rop.replace(" ", "")
            if any(cleaned == f"divide({v},{v})" for v in ["100", "1000", "10", "1"]):
                continue
            non_trivial_ops.append(rop)

        is_complex_multistep_math = len(non_trivial_ops) >= 2
        dependency_depth = max(len(non_trivial_ops), 1)

        # 3. Row Index Inspection for Table Ratio Disambiguation (Rule 2)
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

        is_cross_row_ratio = has_multiple_table_rows or (is_part_to_whole_cue and modality == "table-only")
        is_single_row_temporal = (has_single_table_row and (is_temporal_growth_cue or is_ratio_cue)) or (
            is_temporal_growth_cue and not is_cross_row_ratio and modality == "table-only"
        )

        # 4. Paradigms of LLM Functioning Scoring Profile
        s_interprets = 0.4
        if any(w in q_text for w in ["margin", "diluted", "fair value", "basis points", "ebitda", "amortization"]):
            s_interprets += 0.3
        if is_ratio_cue or is_temporal_growth_cue:
            s_interprets += 0.2
        s_interprets = min(s_interprets, 1.0)

        s_retrieves = 0.3
        if modality == "table-text":
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

        # 5. High-Accuracy Decision Logic Applying Rule 1 & Rule 2
        # Rule 1 Check: Cross-Modal Table + Text
        if modality == "table-text":
            cluster = "Multi-hop QA"
            confidence = 0.98
            sub_category = "Cross-Modal Hybrid Hop (Table + Text)"
            rationale = "Requires synthesizing disclosures across unstructured narrative text and the structured financial table."

        # Rule 1 Check: Dispersed Multi-Sentence Text Navigation (>= 2 sentences)
        elif modality == "text-only" and hop_count >= 2:
            cluster = "Multi-hop QA"
            confidence = 0.88
            sub_category = "Dispersed Multi-Sentence Text Navigation"
            rationale = "Requires retrieving and synthesizing multiple distinct sentences dispersed across narrative disclosure paragraphs."

        # Rule 2 Check: Single-Row Horizontal Temporal Ratio
        elif modality == "table-only" and is_single_row_temporal:
            cluster = "Numerical QA"
            confidence = 0.95
            sub_category = "Single-Row Temporal Ratio"
            rationale = "Evidence is co-located in a single tabular row across reporting periods; cognitive bottleneck is formulating the horizontal percentage or growth rate formula."

        # Rule 2 Check: Cross-Row Vertical Column Ratio (Part-to-Whole)
        elif modality == "table-only" and is_cross_row_ratio:
            if not is_complex_multistep_math:
                cluster = "Numerical QA"
                confidence = 0.96
                sub_category = "Cross-Row Column Ratio (Part-to-Whole)"
                rationale = "Evidence requires retrieving distinct rows (component and total) within the table; classified as Numerical QA because it executes a direct single-step division without intermediate chained dependencies."
            else:
                cluster = "Multi-hop QA"
                confidence = 0.92
                sub_category = "Multi-Row Tabular Bridge"
                rationale = "Requires navigating across multiple distinct table rows combined with multi-step sequential reasoning."

        # Rule 1 Check: Single-Fact (1 Hop) with Direct Arithmetic vs Complex Multi-Step Math
        elif hop_count == 1 and not is_complex_multistep_math:
            cluster = "Numerical QA"
            confidence = 0.92
            sub_category = "Single-Fact Direct Arithmetic"
            rationale = "Evidence is co-located in a single sentence or table row; classified as Numerical QA because the computation is direct single-fact arithmetic (with elementary unit normalization) without multi-step complex mathematics."

        elif hop_count == 1 and is_complex_multistep_math:
            cluster = "Multi-hop QA"
            confidence = 0.85
            sub_category = "Multi-Step Complex Mathematical Reasoning"
            rationale = "Evidence is localized to a single sentence or table row but requires complex multi-step mathematical composition and intermediate state evaluation."

        # Table Row Aggregations
        elif is_aggregation and modality == "table-only":
            cluster = "Numerical QA"
            confidence = 0.94
            sub_category = "Tabular Aggregation (Sum/Avg/Min/Max)"
            rationale = "Direct symbolic aggregation over a localized table segment or row line-items."

        # General Multi-Fact vs Localized
        else:
            if hop_count >= 2 or is_complex_multistep_math:
                cluster = "Multi-hop QA"
                confidence = 0.82
                sub_category = "Multi-Row Tabular Bridge" if modality == "table-only" else "Chained Intermediate Dependency Hop"
                rationale = "Requires multi-fact evidence traversal and sequential dependency resolution."
            else:
                cluster = "Numerical QA"
                confidence = 0.86
                sub_category = "Co-Located Multi-Metric Operation"
                rationale = "All quantitative parameters are co-located in the same immediate context; primary challenge is arithmetic execution."

        return ClassificationResult(
            cluster=cluster,
            sub_category=sub_category,
            confidence=round(confidence, 3),
            evidence_modality=modality,
            hop_count=hop_count,
            dependency_depth=dependency_depth,
            is_chained=is_chained,
            is_aggregation=is_aggregation,
            paradigm_interprets=round(s_interprets, 2),
            paradigm_retrieves=round(s_retrieves, 2),
            paradigm_solves=round(s_solves, 2),
            human_rationale=rationale
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

    # If network fetch fails, provide representative sample instances so dashboard is functional
    if not all_raw:
        sample_mock = [
            {
                "id": "mock_sample_1",
                "_split": "train",
                "pre_text": ["Note 4: Restructuring charges in 2018 were $50 million."],
                "post_text": ["Operating activities provided consistent positive cash flow."],
                "table": [["", "2018", "2017"], ["Total revenue", "1250", "1000"], ["Operating income", "300", "250"]],
                "qa": {"question": "Considering restructuring costs in note 4, what was the adjusted operating margin in 2018?", "program": "add(300, 50), divide(#0, 1250)", "exe_ans": "0.28", "answer": "28%", "gold_inds": {"text_0": "Note 4", "table_1": "1250", "table_2": "300"}}
            },
            {
                "id": "mock_sample_2",
                "_split": "validation",
                "pre_text": ["Financial highlights for the period."],
                "post_text": [],
                "table": [["", "2018", "2017"], ["Total revenue", "1250", "1000"], ["Net profit", "150", "120"]],
                "qa": {"question": "What was the percentage change in total revenue from 2017 to 2018?", "program": "subtract(1250, 1000), divide(#0, 1000)", "exe_ans": "0.25", "answer": "25%", "gold_inds": {"table_1": "Total revenue"}}
            },
            {
                "id": "mock_sample_3",
                "_split": "test",
                "pre_text": [],
                "post_text": [],
                "table": [["Segment", "R&D Expense"], ["Hardware", "250"], ["Software", "400"], ["Services", "350"]],
                "qa": {"question": "What is the sum of R&D expenses across all segments?", "program": "table-sum(R&D Expense)", "exe_ans": "1000", "answer": "1,000", "gold_inds": {"table_0": "250", "table_1": "400", "table_2": "350"}}
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
    st.markdown('<div class="main-title">📊 FinQA LLM Task Clustering Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Systematic Study of All 8,281 Instances across Cognitive Paradigms: <i>Interprets</i>, <i>Retrieves</i>, and <i>Solves</i></div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Dataset & Filters")
        
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

        st.subheader("Filter Instances")
        split_choices = ["All Splits"] + sorted(list(df_clustered['split'].unique()))
        selected_split = st.selectbox("Split:", split_choices, index=0)

        cluster_choices = ["All Clusters", "Multi-hop QA", "Numerical QA"]
        selected_cluster = st.selectbox("Cluster Category:", cluster_choices, index=0)

        modality_choices = ["All Modalities"] + sorted(list(df_clustered['modality'].unique()))
        selected_modality = st.selectbox("Evidence Modality:", modality_choices, index=0)

        search_query = st.text_input("🔍 Search question / ID:", "").strip().lower()

    # Filter DataFrame
    filtered_df = df_clustered.copy()
    if selected_split != "All Splits":
        filtered_df = filtered_df[filtered_df['split'] == selected_split]
    if selected_cluster != "All Clusters":
        filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]
    if selected_modality != "All Modalities":
        filtered_df = filtered_df[filtered_df['modality'] == selected_modality]
    if search_query:
        filtered_df = filtered_df[
            filtered_df['question'].str.lower().str.contains(search_query) | 
            filtered_df['id'].str.lower().str.contains(search_query)
        ]

    matched_indices = list(filtered_df.index)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Instance Explorer",
        "📊 Dataset Analytics & Paradigms",
        "💾 Batch Clustering & Export",
        "📖 Taxonomy & Methodology"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: INSTANCE EXPLORER
    # --------------------------------------------------------------------------
    with tab1:
        if not matched_indices:
            st.warning("No instances match the current filters. Please adjust your criteria.")
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

            # Diagnostic Card
            st.divider()
            c_top1, c_top2 = st.columns([3, 1])
            with c_top1:
                st.subheader(f"Instance: `{current_inst.id}`")
                st.caption(f"Split: **{current_inst.split}** | Match **#{st.session_state.view_pointer + 1:,}** of **{len(matched_indices):,}** | Global Index: **#{active_idx:,}**")
            with c_top2:
                if current_row['cluster'] == "Multi-hop QA":
                    st.markdown('<div style="text-align:right;"><span class="badge-multihop">Multi-hop QA</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="text-align:right;"><span class="badge-numerical">Numerical QA</span></div>', unsafe_allow_html=True)
                st.caption(f"Confidence: **{int(current_row['confidence'] * 100)}%**")

            # Sub-category & Rationale
            st.info(f"**Sub-Category:** {current_row['sub_category']}  \n**Modality:** `{current_row['modality']}` | **Retrieval Hops:** `{current_row['hop_count']}` | **Chained Program:** `{current_row['is_chained']}`  \n**Human Auditor Rationale:** {current_row['rationale']}")

            # Cognitive Load Gauges
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Interprets (Domain Knowledge)", f"{int(current_row['interprets_score'] * 100)}%")
                st.progress(current_row['interprets_score'])
            with m2:
                st.metric("Retrieves (Evidence Nav)", f"{int(current_row['retrieves_score'] * 100)}%")
                st.progress(current_row['retrieves_score'])
            with m3:
                st.metric("Solves (Math Execution)", f"{int(current_row['solves_score'] * 100)}%")
                st.progress(current_row['solves_score'])

            # Question & Answer Box
            st.markdown(f"#### Question❓ :\n> **{current_inst.question}**")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"**Execution Answer:** `{current_inst.final_result or current_inst.answer or 'N/A'}`")
            with col_a2:
                st.markdown(f"**Reasoning Program:** `{current_inst.program or 'N/A'}`")

            # Evidence Footprints (Table and Texts)
            st.markdown("### Context Disclosures & Tables")
            with st.expander("Narrative Pre-Text Disclosures", expanded=bool(current_inst.pre_text)):
                if current_inst.pre_text:
                    for p in current_inst.pre_text:
                        st.markdown(f"- {p}")
                else:
                    st.caption("No pre-text disclosures.")

            with st.expander("Structured Financial Statement Table", expanded=bool(current_inst.table)):
                if current_inst.table and len(current_inst.table) > 1:
                    st.dataframe(current_inst.table_df, use_container_width=True)
                else:
                    st.caption("No table available.")

            with st.expander("Narrative Post-Text Disclosures", expanded=bool(current_inst.post_text)):
                if current_inst.post_text:
                    for p in current_inst.post_text:
                        st.markdown(f"- {p}")
                else:
                    st.caption("No post-text disclosures.")

    # --------------------------------------------------------------------------
    # TAB 2: DATASET ANALYTICS & PARADIGMS
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Global Cognitive Paradigm Distribution")
        
        # High-level Metrics
        total_n = len(df_clustered)
        n_multi = (df_clustered['cluster'] == "Multi-hop QA").sum()
        n_num = (df_clustered['cluster'] == "Numerical QA").sum()
        pct_hybrid = (df_clustered['modality'] == "table-text").mean() * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Instances", f"{total_n:,}")
        k2.metric("Multi-hop QA", f"{n_multi:,}", f"{n_multi/total_n:.1%}")
        k3.metric("Numerical QA", f"{n_num:,}", f"{n_num/total_n:.1%}")
        k4.metric("Cross-Modal (Table+Text)", f"{pct_hybrid:.1f}%")

        st.divider()

        # Visualizations Suite with Isolated Container Layout (Zero Label Overlap)
        palette = {"Multi-hop QA": "#1E88E5", "Numerical QA": "#2E7D32"}

        st.markdown("### Core Distributions & Modality")
        col_top1, col_top2, col_top3 = st.columns(3)

        # 1. Donut Chart
        with col_top1:
            fig1, ax1 = plt.subplots(figsize=(4.8, 3.8))
            c_counts = df_clustered['cluster'].value_counts()
            labels = [f"{k}\n({v:,} | {v/total_n:.1%})" for k, v in c_counts.items()]
            ax1.pie(
                c_counts, labels=labels, startangle=140, 
                colors=[palette.get(k, '#999999') for k in c_counts.index],
                wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
            )
            ax1.set_title(f"1. Primary Clustering (N = {total_n:,})", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig1)

        # 2. Split Stacked Bar
        with col_top2:
            fig2, ax2 = plt.subplots(figsize=(4.8, 3.8))
            split_cross = pd.crosstab(df_clustered['split'], df_clustered['cluster'], normalize='index') * 100
            split_cross.plot(kind='bar', stacked=True, ax=ax2, color=[palette["Multi-hop QA"], palette["Numerical QA"]])
            ax2.set_title("2. Cluster Distribution by Split (%)", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Percentage (%)")
            ax2.set_xlabel("Split")
            ax2.legend(loc='lower right', fontsize=8)
            ax2.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig2)

        # 3. Modality Stacked Bar
        with col_top3:
            fig3, ax3 = plt.subplots(figsize=(4.8, 3.8))
            mod_cross = pd.crosstab(df_clustered['modality'], df_clustered['cluster'], normalize='index') * 100
            mod_cross.plot(kind='bar', stacked=True, ax=ax3, color=[palette["Multi-hop QA"], palette["Numerical QA"]])
            ax3.set_title("3. Modality Distribution (%)", fontsize=11, fontweight='bold')
            ax3.set_ylabel("Percentage (%)")
            ax3.set_xlabel("Modality")
            ax3.legend(loc='lower right', fontsize=8)
            ax3.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            st.pyplot(fig3)

        st.divider()

        st.markdown("### Cognitive Paradigms & Specialized Sub-Categories")
        col_bot1, col_bot2 = st.columns([1, 1.1])

        # 4. Cognitive Load Profiles (Isolated in Left Column)
        with col_bot1:
            fig4, ax4 = plt.subplots(figsize=(6.8, 4.8))
            paradigm_df = df_clustered.melt(
                id_vars=['cluster'],
                value_vars=['interprets_score', 'retrieves_score', 'solves_score'],
                var_name='Paradigm', value_name='Cognitive Load'
            )
            paradigm_map = {
                'interprets_score': 'Interprets\n(Domain Jargon)',
                'retrieves_score': 'Retrieves\n(Evidence Nav)',
                'solves_score': 'Solves\n(Math Execution)'
            }
            paradigm_df['Paradigm'] = paradigm_df['Paradigm'].map(paradigm_map)
            sns.boxplot(
                data=paradigm_df, x='Paradigm', y='Cognitive Load', 
                hue='cluster', ax=ax4, palette=palette
            )
            ax4.set_title("4. LLM Paradigm Cognitive Load Profiles", fontsize=11, fontweight='bold')
            ax4.set_ylim(0, 1.1)
            ax4.legend(loc='upper right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig4)

        # 5. Top Sub-categories (Isolated in Right Column with Wrapped Labels)
        with col_bot2:
            fig5, ax5 = plt.subplots(figsize=(7.4, 4.8))
            sub_counts = df_clustered['sub_category'].value_counts().head(6)
            sub_df = df_clustered[df_clustered['sub_category'].isin(sub_counts.index)].copy()
            # Wrap long text labels to prevent horizontal bleed
            sub_df['sub_wrapped'] = sub_df['sub_category'].apply(lambda x: textwrap.fill(x, 24))
            wrapped_order = [textwrap.fill(x, 24) for x in sub_counts.index]
            sns.countplot(
                data=sub_df, y='sub_wrapped', hue='cluster', 
                ax=ax5, palette=palette, order=wrapped_order
            )
            ax5.set_title("5. Granular Sub-Categories", fontsize=11, fontweight='bold')
            ax5.set_xlabel("Count")
            ax5.set_ylabel("")
            ax5.legend(loc='lower right', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig5)

        # Summary Data Table
        st.markdown("### Cross-Tabulation Summary")
        summary_tbl = df_clustered.groupby(['split', 'cluster']).size().unstack(fill_value=0)
        summary_tbl['Total'] = summary_tbl.sum(axis=1)
        st.table(summary_tbl)

    # --------------------------------------------------------------------------
    # TAB 3: BATCH CLUSTERING & EXPORT
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Export Clustered Dataset")
        st.write(f"The table below contains all **{len(df_clustered):,}** analyzed instances ready for export.")

        st.dataframe(df_clustered, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Dataset as CSV",
                data=csv_data,
                file_name="finqa_all_clustered_instances.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            jsonl_data = df_clustered.to_json(orient="records", lines=True).encode('utf-8')
            st.download_button(
                label="Download Dataset as JSONL",
                data=jsonl_data,
                file_name="finqa_all_clustered_instances.jsonl",
                mime="application/json",
                use_container_width=True
            )

    # --------------------------------------------------------------------------
    # TAB 4: TAXONOMY & METHODOLOGY
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Theoretical Taxonomy: The 3 Paradigms of LLM Functioning")
        st.markdown("""
        When processing financial earnings statements, an LLM operates across three core functional paradigms:
        
        1. **Interprets (Semantic & Accounting Grounding)**:
           - Deconstructs domain terminology (*"operating margin"*, *"diluted EPS"*, *"fair value of stock options"*, *"basis points"*).
           - Formulates implicit financial equations (e.g., Growth Rate $= \\frac{\\text{Current} - \\text{Prior}}{\\text{Prior}}$).
           - Resolves temporal references across reporting periods.

        2. **Retrieves (Multi-Modal Context Navigation)**:
           - Scans semi-structured tables and surrounding unstructured disclosures.
           - Determines whether evidence is co-located in a single sentence/row or dispersed across boundaries.
           - Connects entity mentions across footnotes and tabular statements.

        3. **Solves (Symbolic & Algorithmic Execution)**:
           - Generates and evaluates executable DSL programs (`add`, `subtract`, `multiply`, `divide`, `exp`, `greater`).
           - Applies row aggregations (`table-sum`, `table-average`, `table-max`, `table-min`).
           - Maintains precision and scales units across millions, thousands, and percentages.
        """)

        st.divider()
        st.subheader("Human-Calibrated Categorization Matrix")
        st.markdown("""
        | Dimension | Multi-hop QA | Numerical QA |
        | :--- | :--- | :--- |
        | **Primary Bottleneck** | **Evidence Navigation & Linking** (*Retrieves* + *Interprets*) | **Calculation & Precision** (*Solves*) |
        | **Evidence Modality** | Cross-modal (`Table + Text`) or dispersed across separated sentences/rows | Localized strictly to a single table row or a single text sentence |
        | **Navigation Topology** | Multi-hop: Traversing question entity $\\rightarrow$ footnote disclosure $\\rightarrow$ table cell | Single-hop: Direct point lookup of adjacent cells or figures |
        | **Program Chaining** | Compositional dependency ($f(g(x), y)$ where $g(x)$ is an intermediate hop) | Direct formula evaluation ($f(x, y)$ or row aggregation) |
        | **Representative Sub-Types** | Cross-modal adjustments, cross-segment reconciliations, multi-paragraph text synthesis | Year-over-year row percentage change, segment sum/average, operating margin ratios |
        """)


if __name__ == "__main__":
    main()
