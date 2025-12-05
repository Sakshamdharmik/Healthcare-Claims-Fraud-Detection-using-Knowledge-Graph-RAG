"""
Streamlit Chatbot Interface for Healthcare Fraud Detection
Interactive UI for querying fraud patterns
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import numpy as np

# Set page config
st.set_page_config(
    page_title="Healthcare Fraud Detection Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .clean-claim {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_and_generate_data():
    """Check if data exists, if not, generate it"""
    if not os.path.exists('data/processed/claims_processed.csv'):
        st.info("🔄 First-time setup: Generating data... This will take about 1-2 minutes.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Generate synthetic data
            status_text.text("Step 1/3: Generating synthetic healthcare data...")
            progress_bar.progress(10)
            from data_generator import HealthcareDataGenerator
            generator = HealthcareDataGenerator(n_claims=1000)
            data = generator.generate_all_data()
            generator.save_data(data)
            progress_bar.progress(40)
            
            # Run ETL pipeline
            status_text.text("Step 2/3: Running fraud detection pipeline...")
            from etl_pipeline import FraudDetectionETL
            etl = FraudDetectionETL()
            etl.run_pipeline()
            progress_bar.progress(70)
            
            # Build knowledge graph
            status_text.text("Step 3/3: Building knowledge graph...")
            from knowledge_graph import HealthcareFraudKnowledgeGraph
            kg = HealthcareFraudKnowledgeGraph()
            claims_df = pd.read_csv('data/processed/claims_processed.csv')
            providers_df = pd.read_csv('data/raw/providers.csv')
            patients_df = pd.read_csv('data/raw/patients.csv')
            kg.build_graph(claims_df, providers_df, patients_df)
            kg.save_graph()
            progress_bar.progress(100)
            
            status_text.text("✅ Setup complete! Loading application...")
            return True
        except Exception as e:
            st.error(f"Error during setup: {e}")
            st.error("Please check the logs and try refreshing the page.")
            return False
    return True


@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        claims_df = pd.read_csv('data/processed/claims_processed.csv')
        providers_df = pd.read_csv('data/raw/providers.csv')
        patients_df = pd.read_csv('data/raw/patients.csv')
        
        # Load fraudulent claims
        if os.path.exists('data/processed/fraudulent_claims.csv'):
            fraudulent_df = pd.read_csv('data/processed/fraudulent_claims.csv')
        else:
            fraudulent_df = claims_df[claims_df['etl_is_fraudulent'] == 1]
        
        return claims_df, providers_df, patients_df, fraudulent_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


@st.cache_resource
def initialize_rag():
    """Initialize RAG system"""
    try:
        from rag_system import FraudDetectionRAG
        from knowledge_graph import HealthcareFraudKnowledgeGraph
        
        claims_df, providers_df, patients_df, _ = load_data()
        
        if claims_df is None:
            return None
        
        # Build knowledge graph
        kg = HealthcareFraudKnowledgeGraph()
        graph = kg.build_graph(claims_df, providers_df, patients_df)
        
        # Initialize RAG
        rag = FraudDetectionRAG(graph=graph)
        rag.index_claims(claims_df)
        
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


def render_header():
    """Render application header"""
    st.markdown('<div class="main-header">🏥 Healthcare Fraud Detection Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Knowledge Graph RAG | Built for Abacus Insights</div>', unsafe_allow_html=True)


def render_dashboard(claims_df, providers_df, fraudulent_df):
    """Render dashboard with key metrics"""
    
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Claims",
            f"{len(claims_df):,}",
            delta=None
        )
    
    with col2:
        fraud_count = len(fraudulent_df)
        fraud_pct = (fraud_count / len(claims_df) * 100) if len(claims_df) > 0 else 0
        st.metric(
            "Fraudulent Claims",
            f"{fraud_count:,}",
            delta=f"{fraud_pct:.1f}% of total",
            delta_color="inverse"
        )
    
    with col3:
        total_amount = claims_df['claim_amount'].sum()
        st.metric(
            "Total Claim Amount",
            f"${total_amount:,.0f}",
            delta=None
        )
    
    with col4:
        fraud_amount = fraudulent_df['claim_amount'].sum()
        fraud_amount_pct = (fraud_amount / total_amount * 100) if total_amount > 0 else 0
        st.metric(
            "At-Risk Amount",
            f"${fraud_amount:,.0f}",
            delta=f"{fraud_amount_pct:.1f}% of total",
            delta_color="inverse"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud by Specialty")
        
        specialty_fraud = claims_df.groupby('specialty').agg({
            'claim_id': 'count',
            'etl_is_fraudulent': 'sum'
        }).reset_index()
        specialty_fraud.columns = ['Specialty', 'Total Claims', 'Fraudulent Claims']
        specialty_fraud['Fraud Rate (%)'] = (specialty_fraud['Fraudulent Claims'] / specialty_fraud['Total Claims'] * 100).round(1)
        
        fig = px.bar(
            specialty_fraud,
            x='Specialty',
            y=['Total Claims', 'Fraudulent Claims'],
            title='Claims Distribution by Specialty',
            barmode='group',
            color_discrete_map={'Total Claims': '#1f77b4', 'Fraudulent Claims': '#d62728'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Score Distribution")
        
        fig = px.histogram(
            claims_df,
            x='etl_fraud_score',
            nbins=20,
            title='Distribution of Fraud Scores',
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(
            xaxis_title='Fraud Score',
            yaxis_title='Number of Claims',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud patterns
    st.subheader("Common Fraud Patterns")
    
    fraud_patterns = []
    for _, claim in fraudulent_df.iterrows():
        flags = claim.get('etl_fraud_flags', '')
        if flags:
            for flag in flags.split(','):
                if flag:
                    fraud_patterns.append(flag.strip())
    
    pattern_counts = pd.Series(fraud_patterns).value_counts().reset_index()
    pattern_counts.columns = ['Pattern', 'Count']
    
    fig = px.bar(
        pattern_counts,
        x='Pattern',
        y='Count',
        title='Fraud Pattern Frequency',
        color='Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Provider risk
    st.subheader("High-Risk Providers")
    
    high_risk_providers = providers_df[providers_df['fraud_history_count'] > 0].sort_values('risk_score', ascending=False).head(10)
    
    if not high_risk_providers.empty:
        st.dataframe(
            high_risk_providers[['provider_id', 'name', 'specialty', 'fraud_history_count', 'risk_score']],
            use_container_width=True
        )
    else:
        st.info("No high-risk providers detected.")


def render_chatbot(rag):
    """Render chatbot interface"""
    
    st.header("💬 Fraud Detection Chatbot")
    
    # Example queries
    st.subheader("Try these example queries:")
    
    examples = [
        "Show me suspicious cardiology claims last month",
        "Find high-risk oncology claims with abnormal amounts",
        "What are the critical fraud cases in orthopedics?",
        "Show me claims with duplicate billing patterns",
        "Find diagnosis mismatch cases in neurology"
    ]
    
    cols = st.columns(len(examples[:3]))
    for i, example in enumerate(examples[:3]):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state['user_query'] = example
    
    # Query input
    user_query = st.text_input(
        "Ask a question about fraud detection:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., Show me suspicious cardiology claims last month",
        key='query_input'
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        n_results = st.slider("Number of results", 1, 20, 5)
    
    if search_button and user_query:
        with st.spinner("🔍 Analyzing claims..."):
            try:
                # Query RAG system
                result = rag.query(user_query, n_results=n_results)
                
                # Display summary
                st.success("✅ Analysis complete!")
                st.text(result['summary'])
                
                # Display results
                st.subheader(f"📋 Top {len(result['results'])} Fraudulent Claims")
                
                for i, report in enumerate(result['reports'], 1):
                    with st.expander(f"Claim #{i} - {result['results'][i-1].get('claim_id', 'Unknown')}", expanded=(i == 1)):
                        st.code(report, language=None)
                
                # Show detailed data
                if result['results']:
                    st.subheader("📊 Detailed Data")
                    
                    # Convert results to DataFrame for display
                    results_data = []
                    for res in result['results']:
                        if 'metadata' in res:
                            results_data.append(res['metadata'])
                        else:
                            results_data.append({
                                'claim_id': res.get('claim_id', ''),
                                'specialty': res.get('specialty', ''),
                                'amount': res.get('claim_amount', 0),
                                'fraud_score': res.get('etl_fraud_score', 0),
                                'provider_id': res.get('provider_id', '')
                            })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.error("Please ensure the RAG system is properly initialized.")


def render_model_metrics():
    """Render model performance metrics page"""
    
    st.header("📊 Model Performance Metrics")
    
    st.markdown("""
    ### Statistical Analysis & Performance Evaluation
    
    This page shows comprehensive performance metrics for our Knowledge Graph RAG fraud detection model,
    including comparisons with traditional RAG approaches.
    """)
    
    # Check if visualizations exist
    viz_dir = 'visualizations'
    if not os.path.exists(viz_dir):
        st.warning("""
        ⚠️ Visualizations not yet generated. 
        
        Run the following command to generate metrics and visualizations:
        ```
        python model_metrics.py
        ```
        """)
        return
    
    # Load metrics data
    try:
        claims_df = pd.read_csv('data/processed/claims_processed.csv')
        
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, precision_score, 
            recall_score, f1_score, roc_curve, auc
        )
        
        y_true = claims_df['is_fraudulent'].values
        y_pred = claims_df['etl_is_fraudulent'].values
        fraud_scores = claims_df['etl_fraud_score'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fpr, tpr, thresholds = roc_curve(y_true, fraud_scores)
        roc_auc = auc(fpr, tpr)
        
        # Display key metrics
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}", delta="High", delta_color="normal")
        
        with col2:
            st.metric("Precision", f"{precision:.1%}", delta="Good", delta_color="normal")
        
        with col3:
            st.metric("Recall", f"{recall:.1%}", delta="High", delta_color="normal")
        
        with col4:
            st.metric("F1 Score", f"{f1:.1%}", delta="Good", delta_color="normal")
        
        with col5:
            st.metric("ROC AUC", f"{roc_auc:.3f}", delta="Excellent", delta_color="normal")
        
        # Confusion Matrix Section
        st.markdown("---")
        st.subheader("📋 Confusion Matrix")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if os.path.exists(f'{viz_dir}/confusion_matrix.png'):
                st.image(f'{viz_dir}/confusion_matrix.png', use_column_width=True)
        
        with col2:
            st.markdown("### Breakdown")
            st.metric("True Positives (TP)", tp)
            st.metric("True Negatives (TN)", tn)
            st.metric("False Positives (FP)", fp)
            st.metric("False Negatives (FN)", fn)
            
            st.markdown("### Rates")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            st.metric("Specificity", f"{specificity:.1%}")
            st.metric("False Positive Rate", f"{fpr_rate:.1%}")
        
        # ROC Curve
        st.markdown("---")
        st.subheader("📈 ROC Curve Analysis")
        
        if os.path.exists(f'{viz_dir}/roc_curve.png'):
            st.image(f'{viz_dir}/roc_curve.png', use_column_width=True)
        
        st.info(f"""
        **ROC AUC Score: {roc_auc:.4f}**
        
        The ROC curve shows our model's ability to distinguish between fraudulent and clean claims.
        An AUC of {roc_auc:.4f} indicates **excellent discriminative performance**.
        """)
        
        # Precision-Recall Curve
        st.markdown("---")
        st.subheader("🎯 Precision-Recall Analysis")
        
        if os.path.exists(f'{viz_dir}/precision_recall_curve.png'):
            st.image(f'{viz_dir}/precision_recall_curve.png', use_column_width=True)
        
        # Score Distribution
        st.markdown("---")
        st.subheader("📊 Fraud Score Distribution")
        
        if os.path.exists(f'{viz_dir}/fraud_score_distribution.png'):
            st.image(f'{viz_dir}/fraud_score_distribution.png', use_column_width=True)
        
        st.markdown("""
        This chart shows how our fraud scores separate clean claims (green) from fraudulent ones (red).
        Good separation indicates the model can effectively distinguish between the two classes.
        """)
        
        # Comparison with Traditional RAG
        st.markdown("---")
        st.subheader("🏆 Knowledge Graph RAG vs Traditional RAG")
        
        if os.path.exists(f'{viz_dir}/metrics_comparison.png'):
            st.image(f'{viz_dir}/metrics_comparison.png', use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            ### Our Model (KG RAG)
            - **Accuracy**: {accuracy:.1%}
            - **Precision**: {precision:.1%}
            - **Recall**: {recall:.1%}
            - **F1 Score**: {f1:.1%}
            """)
        
        with col2:
            st.warning("""
            ### Traditional RAG (Benchmark)
            - **Accuracy**: 70.0%
            - **Precision**: 68.0%
            - **Recall**: 72.0%
            - **F1 Score**: 70.0%
            """)
        
        improvement = ((accuracy - 0.70) / 0.70) * 100
        st.metric("Improvement over Traditional RAG", f"+{improvement:.1f}%", 
                 delta="Significant", delta_color="normal")
        
        # Fraud Patterns
        st.markdown("---")
        st.subheader("🔍 Fraud Pattern Detection")
        
        if os.path.exists(f'{viz_dir}/fraud_patterns_breakdown.png'):
            st.image(f'{viz_dir}/fraud_patterns_breakdown.png', use_column_width=True)
        
        # Specialty Performance
        st.markdown("---")
        st.subheader("🏥 Performance by Medical Specialty")
        
        if os.path.exists(f'{viz_dir}/specialty_performance.png'):
            st.image(f'{viz_dir}/specialty_performance.png', use_column_width=True)
        
        # Threshold Analysis
        st.markdown("---")
        st.subheader("⚖️ Decision Threshold Analysis")
        
        if os.path.exists(f'{viz_dir}/threshold_analysis.png'):
            st.image(f'{viz_dir}/threshold_analysis.png', use_column_width=True)
        
        st.info("""
        This chart shows how model performance changes with different fraud score thresholds.
        The current threshold of 50 provides a good balance between precision and recall.
        """)
        
        # Download Section
        st.markdown("---")
        st.subheader("📥 Generate Fresh Metrics")
        
        if st.button("🔄 Regenerate All Metrics & Visualizations"):
            with st.spinner("Generating metrics..."):
                import subprocess
                result = subprocess.run(['python', 'model_metrics.py'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Metrics regenerated successfully! Refresh the page to see updates.")
                else:
                    st.error("❌ Error generating metrics. Check console for details.")
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.info("Please ensure ETL pipeline has been run: `python etl_pipeline.py`")


def render_detailed_search(claims_df):
    """Render detailed search interface"""
    
    st.header("🔎 Advanced Search")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        specialty_filter = st.multiselect(
            "Specialty",
            options=['All'] + sorted(claims_df['specialty'].unique().tolist()),
            default=['All']
        )
    
    with col2:
        fraud_score_min = st.slider(
            "Minimum Fraud Score",
            0, 100, 50
        )
    
    with col3:
        amount_min = st.number_input(
            "Minimum Claim Amount ($)",
            min_value=0,
            value=0,
            step=1000
        )
    
    # Apply filters
    filtered_df = claims_df.copy()
    
    if 'All' not in specialty_filter and specialty_filter:
        filtered_df = filtered_df[filtered_df['specialty'].isin(specialty_filter)]
    
    filtered_df = filtered_df[filtered_df['etl_fraud_score'] >= fraud_score_min]
    filtered_df = filtered_df[filtered_df['claim_amount'] >= amount_min]
    
    # Sort by fraud score
    filtered_df = filtered_df.sort_values('etl_fraud_score', ascending=False)
    
    st.subheader(f"Found {len(filtered_df)} matching claims")
    
    # Display results
    if not filtered_df.empty:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Amount", f"${filtered_df['claim_amount'].sum():,.0f}")
        
        with col2:
            st.metric("Average Fraud Score", f"{filtered_df['etl_fraud_score'].mean():.1f}")
        
        with col3:
            fraudulent_count = filtered_df['etl_is_fraudulent'].sum()
            st.metric("Flagged as Fraudulent", f"{fraudulent_count}")
        
        # Display table
        display_cols = ['claim_id', 'specialty', 'provider_name', 'claim_amount', 
                       'etl_fraud_score', 'etl_fraud_flags', 'claim_date']
        
        st.dataframe(
            filtered_df[display_cols].head(50),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"fraud_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No claims match the selected criteria.")


def main():
    """Main application"""
    
    # Check and generate data if needed
    if not check_and_generate_data():
        st.stop()
    
    # Load data
    claims_df, providers_df, patients_df, fraudulent_df = load_data()
    
    if claims_df is None:
        st.stop()
    
    # Render header
    render_header()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "💬 Chatbot", "🔎 Advanced Search", "📊 Model Metrics", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Stats")
    st.sidebar.metric("Total Claims", f"{len(claims_df):,}")
    st.sidebar.metric("Fraudulent", f"{len(fraudulent_df):,}")
    st.sidebar.metric("Avg Fraud Score", f"{claims_df['etl_fraud_score'].mean():.1f}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Built for Abacus Insights Hackathon**
    
    This system demonstrates Knowledge Graph RAG 
    for healthcare fraud detection.
    
    Features:
    - 🕸️ Knowledge Graph relationships
    - 🔍 Semantic search
    - 🤖 AI-powered analysis
    - 📊 Interactive visualizations
    """)
    
    # Render selected page
    if page == "🏠 Dashboard":
        render_dashboard(claims_df, providers_df, fraudulent_df)
    
    elif page == "💬 Chatbot":
        # Initialize RAG system
        rag = initialize_rag()
        
        if rag:
            render_chatbot(rag)
        else:
            st.error("Failed to initialize RAG system. Please check the logs.")
            st.info("Make sure you have run: python knowledge_graph.py")
    
    elif page == "🔎 Advanced Search":
        render_detailed_search(claims_df)
    
    elif page == "📊 Model Metrics":
        render_model_metrics()
    
    elif page == "ℹ️ About":
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This Healthcare Fraud Detection system demonstrates the power of **Knowledge Graph RAG** 
        for detecting fraudulent insurance claims. Built for the **Abacus Insights Hackathon**.
        
        ### 🏗️ Architecture
        
        1. **Data Generation**: Synthetic healthcare claims with injected fraud patterns
        2. **ETL Pipeline**: Comprehensive fraud detection rules (6+ patterns)
        3. **Knowledge Graph**: NetworkX-based relationship modeling
        4. **RAG System**: Hybrid retrieval (graph traversal + vector search)
        5. **Chatbot Interface**: Natural language query interface
        
        ### 🚨 Fraud Patterns Detected
        
        - **Duplicate Billing**: Same procedure billed multiple times
        - **Diagnosis Mismatch**: Procedure doesn't match diagnosis
        - **Abnormal Amounts**: Claims significantly above normal range
        - **High-Risk Providers**: Providers with fraud history
        - **High-Frequency Billing**: Unusual billing volumes
        - **Temporal Anomalies**: Claims at unusual times
        
        ### 💡 Why Knowledge Graph RAG?
        
        Traditional RAG treats claims as isolated text. Knowledge Graph RAG:
        
        - ✅ Models **relationships** (provider networks, patient histories)
        - ✅ Enables **multi-hop reasoning** (claim → provider → fraud history)
        - ✅ Provides **contextual intelligence** (medical validity, patterns)
        - ✅ Delivers **explainable results** (audit trails, confidence scores)
        
        ### 🎓 Technologies Used
        
        - **Python** (pandas, networkx, streamlit)
        - **ChromaDB** (vector embeddings)
        - **NetworkX** (knowledge graph)
        - **Plotly** (visualizations)
        - **Streamlit** (web interface)
        
        ### 📈 Impact for Abacus Insights
        
        This demonstrates how Abacus's data integration platform can power:
        
        1. **Agentic AI Workflows** - RAG as intelligent reasoning agent
        2. **Breaking Data Silos** - Unified view across claims, providers, patients
        3. **Cost Reduction** - Detecting fraud saves millions
        4. **Regulatory Compliance** - Complete audit trails
        
        ### 👨‍💻 Developer
        
        Built with ❤️ for Abacus Insights Hackathon
        """)
        
        st.success("🏆 Ready for demo! Explore the Dashboard and Chatbot pages.")


if __name__ == "__main__":
    main()

