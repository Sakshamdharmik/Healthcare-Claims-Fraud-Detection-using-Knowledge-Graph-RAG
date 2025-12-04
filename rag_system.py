"""
RAG System for Healthcare Fraud Detection
Hybrid retrieval using Knowledge Graph + Vector Search
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Run: pip install chromadb")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import networkx as nx


class FraudDetectionRAG:
    """RAG system combining Knowledge Graph and Vector Search"""
    
    def __init__(self, graph: nx.MultiDiGraph = None, use_local_embeddings: bool = True):
        self.graph = graph
        self.use_local_embeddings = use_local_embeddings
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            try:
                self.chroma_client.delete_collection("fraud_claims")
            except:
                pass
            self.collection = self.chroma_client.create_collection(
                name="fraud_claims",
                metadata={"description": "Healthcare fraud claims"}
            )
        else:
            self.collection = None
        
        # Initialize OpenAI client (optional)
        self.openai_client = None
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI()
        
        self.claims_data = None
        
    def index_claims(self, claims_df: pd.DataFrame):
        """Index claims into vector database"""
        print("📚 Indexing claims for semantic search...")
        
        self.claims_data = claims_df
        
        if not CHROMADB_AVAILABLE:
            print("   ⚠️  ChromaDB not available, skipping indexing")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for _, claim in claims_df.iterrows():
            # Create rich text representation
            fraud_flags = claim.get('etl_fraud_flags', '')
            fraud_score = claim.get('etl_fraud_score', 0)
            
            doc_text = f"""
            Claim {claim['claim_id']} - {claim['specialty']} 
            Provider: {claim['provider_name']} ({claim['provider_id']})
            Patient: {claim['patient_id']}
            Procedure: {claim['procedure_name']} (CPT {claim['cpt_code']})
            Diagnosis: {claim['diagnosis_name']} (ICD {claim['icd_code']})
            Amount: ${claim['claim_amount']:.2f}
            Date: {claim['claim_date']}
            Fraud Score: {fraud_score}
            Fraud Flags: {fraud_flags if fraud_flags else 'None'}
            Status: {'FRAUDULENT' if claim.get('etl_is_fraudulent', 0) == 1 else 'Clean'}
            """
            
            documents.append(doc_text.strip())
            
            metadatas.append({
                'claim_id': str(claim['claim_id']),
                'specialty': str(claim['specialty']),
                'provider_id': str(claim['provider_id']),
                'fraud_score': int(fraud_score),
                'is_fraudulent': int(claim.get('etl_is_fraudulent', 0)),
                'amount': float(claim['claim_amount']),
                'date': str(claim['claim_date'])
            })
            
            ids.append(str(claim['claim_id']))
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"   ✅ Indexed {len(documents)} claims")
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured filters"""
        query_lower = query.lower()
        
        filters = {
            'specialty': None,
            'time_period': None,
            'fraud_threshold': None,
            'keywords': []
        }
        
        # Detect specialty
        specialties = ['cardiology', 'oncology', 'orthopedics', 'neurology', 
                      'dermatology', 'primary care', 'emergency']
        for specialty in specialties:
            if specialty in query_lower:
                filters['specialty'] = specialty.title()
                if specialty == 'primary care':
                    filters['specialty'] = 'Primary Care'
                elif specialty == 'emergency':
                    filters['specialty'] = 'Emergency Medicine'
        
        # Detect time period
        time_keywords = {
            'last month': 30,
            'this month': 30,
            'last week': 7,
            'yesterday': 1,
            'last quarter': 90,
            'last 6 months': 180
        }
        
        for keyword, days in time_keywords.items():
            if keyword in query_lower:
                filters['time_period'] = days
                break
        
        # Detect fraud-related keywords
        if any(word in query_lower for word in ['suspicious', 'fraud', 'high-risk', 'flagged']):
            filters['fraud_threshold'] = 50
        
        if 'critical' in query_lower or 'severe' in query_lower:
            filters['fraud_threshold'] = 70
        
        # Extract keywords
        fraud_keywords = ['duplicate', 'billing', 'mismatch', 'abnormal', 'upcoding']
        filters['keywords'] = [kw for kw in fraud_keywords if kw in query_lower]
        
        return filters
    
    def retrieve_from_graph(self, filters: Dict[str, Any]) -> List[Dict]:
        """Retrieve claims using graph traversal"""
        if not self.graph:
            return []
        
        results = []
        
        # Traverse graph based on filters
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'Claim':
                # Apply filters
                if filters['fraud_threshold'] and node_data.get('fraud_score', 0) < filters['fraud_threshold']:
                    continue
                
                # Get connected provider to check specialty
                if filters['specialty']:
                    provider_found = False
                    for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                        if self.graph.nodes[target].get('node_type') == 'Provider':
                            if self.graph.nodes[target].get('specialty') == filters['specialty']:
                                provider_found = True
                                break
                    
                    if not provider_found:
                        continue
                
                results.append(node_data)
        
        return results
    
    def retrieve_from_vector_db(self, query: str, filters: Dict[str, Any], n_results: int = 10) -> List[Dict]:
        """Retrieve claims using semantic search"""
        if not CHROMADB_AVAILABLE or not self.collection:
            # Fallback to dataframe filtering
            return self._fallback_retrieval(filters, n_results)
        
        # Build where clause for ChromaDB
        where_clause = {}
        
        if filters['fraud_threshold']:
            where_clause['fraud_score'] = {'$gte': filters['fraud_threshold']}
        
        if filters['specialty']:
            where_clause['specialty'] = filters['specialty']
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Convert to list of dicts
            retrieved_claims = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, claim_id in enumerate(results['ids'][0]):
                    retrieved_claims.append({
                        'claim_id': claim_id,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
            
            return retrieved_claims
        except Exception as e:
            print(f"   ⚠️  Vector search failed: {e}")
            return self._fallback_retrieval(filters, n_results)
    
    def _fallback_retrieval(self, filters: Dict[str, Any], n_results: int = 10) -> List[Dict]:
        """Fallback retrieval using pandas filtering"""
        if self.claims_data is None:
            return []
        
        df = self.claims_data.copy()
        
        # Apply filters
        if filters['specialty']:
            df = df[df['specialty'] == filters['specialty']]
        
        if filters['fraud_threshold']:
            df = df[df['etl_fraud_score'] >= filters['fraud_threshold']]
        
        # Sort by fraud score
        df = df.sort_values('etl_fraud_score', ascending=False)
        
        # Take top n
        df = df.head(n_results)
        
        return df.to_dict('records')
    
    def generate_fraud_report(self, claim_data: Dict, query: str) -> str:
        """Generate detailed fraud report for a claim"""
        
        # Extract claim details
        if 'metadata' in claim_data:
            # From vector DB
            metadata = claim_data['metadata']
            claim_id = metadata['claim_id']
            fraud_score = metadata['fraud_score']
            specialty = metadata['specialty']
            amount = metadata['amount']
        else:
            # From direct DataFrame
            claim_id = claim_data.get('claim_id', 'Unknown')
            fraud_score = claim_data.get('etl_fraud_score', 0)
            specialty = claim_data.get('specialty', 'Unknown')
            amount = claim_data.get('claim_amount', 0)
        
        # Get full claim details from dataframe
        if self.claims_data is not None:
            full_claim = self.claims_data[self.claims_data['claim_id'] == claim_id]
            if not full_claim.empty:
                claim_record = full_claim.iloc[0]
            else:
                return f"Claim {claim_id} not found in database."
        else:
            claim_record = claim_data
        
        # Build comprehensive report
        report = self._build_detailed_report(claim_record)
        
        return report
    
    def _build_detailed_report(self, claim: pd.Series) -> str:
        """Build detailed fraud report"""
        
        fraud_flags = claim.get('etl_fraud_flags', '')
        flag_list = [f.strip() for f in fraud_flags.split(',')] if fraud_flags else []
        
        # Determine risk level
        fraud_score = claim.get('etl_fraud_score', 0)
        if fraud_score >= 80:
            risk_level = "🔴 CRITICAL"
        elif fraud_score >= 60:
            risk_level = "🟠 HIGH"
        elif fraud_score >= 40:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        
        report = f"""
{'='*70}
🚨 FRAUD DETECTION REPORT
{'='*70}

CLAIM INFORMATION
{'─'*70}
Claim ID:           {claim['claim_id']}
Risk Level:         {risk_level}
Fraud Score:        {fraud_score}/100
Status:             {'⚠️ FLAGGED AS FRAUDULENT' if claim.get('etl_is_fraudulent', 0) == 1 else '✓ Clean'}

FINANCIAL DETAILS
{'─'*70}
Claim Amount:       ${claim['claim_amount']:,.2f}
Claim Date:         {claim['claim_date']}

MEDICAL DETAILS
{'─'*70}
Specialty:          {claim['specialty']}
Procedure:          {claim['procedure_name']} (CPT {claim['cpt_code']})
Diagnosis:          {claim['diagnosis_name']} (ICD {claim['icd_code']})

PROVIDER INFORMATION
{'─'*70}
Provider ID:        {claim['provider_id']}
Provider Name:      {claim['provider_name']}
"""
        
        # Add fraud indicators
        if flag_list:
            report += f"\n{'─'*70}\n"
            report += "FRAUD INDICATORS DETECTED\n"
            report += f"{'─'*70}\n"
            
            for i, flag in enumerate(flag_list, 1):
                flag_desc = self._get_flag_description(flag)
                report += f"{i}. {flag_desc}\n"
        
        # Add provider risk information
        if claim.get('fraud_history_count', 0) > 0:
            report += f"\n{'─'*70}\n"
            report += "PROVIDER RISK FACTORS\n"
            report += f"{'─'*70}\n"
            report += f"Previous Fraud Incidents:  {claim.get('fraud_history_count', 0)}\n"
            report += f"Provider Risk Score:       {claim.get('risk_score', 0)}\n"
        
        # Add recommendations
        report += f"\n{'─'*70}\n"
        report += "RECOMMENDED ACTIONS\n"
        report += f"{'─'*70}\n"
        
        if fraud_score >= 70:
            report += "☐ IMMEDIATE: Flag for senior auditor review\n"
            report += "☐ IMMEDIATE: Suspend payment pending investigation\n"
            report += "☐ Request complete medical records\n"
            report += "☐ Contact patient to verify procedure\n"
        elif fraud_score >= 50:
            report += "☐ Flag for routine audit review\n"
            report += "☐ Request supporting documentation\n"
            report += "☐ Monitor provider for patterns\n"
        else:
            report += "☐ Standard processing\n"
            report += "☐ Routine monitoring\n"
        
        report += f"\n{'='*70}\n"
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*70}\n"
        
        return report
    
    def _get_flag_description(self, flag: str) -> str:
        """Get human-readable description of fraud flag"""
        descriptions = {
            'duplicate': '⚠️  DUPLICATE BILLING - Same procedure billed multiple times',
            'abnormal_amount': '💰 ABNORMAL AMOUNT - Claim amount significantly exceeds normal range',
            'mismatch': '🩺 DIAGNOSIS MISMATCH - Procedure does not match diagnosis code',
            'high_frequency': '📊 HIGH FREQUENCY - Unusual billing volume detected',
            'high_risk_provider': '👨‍⚕️ HIGH-RISK PROVIDER - Provider has fraud history',
            'temporal_anomaly': '⏰ TEMPORAL ANOMALY - Claim submitted at unusual time'
        }
        
        return descriptions.get(flag, f'⚠️  {flag.upper().replace("_", " ")}')
    
    def query(self, user_query: str, n_results: int = 5) -> Dict[str, Any]:
        """Main query interface - combines graph and vector search"""
        
        print(f"\n🔍 Processing query: '{user_query}'")
        
        # Parse query
        filters = self.parse_query(user_query)
        print(f"   📝 Extracted filters: {filters}")
        
        # Retrieve from vector DB
        vector_results = self.retrieve_from_vector_db(user_query, filters, n_results)
        print(f"   ✅ Retrieved {len(vector_results)} results from vector search")
        
        # Generate reports
        reports = []
        for result in vector_results[:n_results]:
            report = self.generate_fraud_report(result, user_query)
            reports.append(report)
        
        # Generate summary
        summary = self._generate_summary(vector_results, filters)
        
        return {
            'query': user_query,
            'filters': filters,
            'results_count': len(vector_results),
            'results': vector_results,
            'reports': reports,
            'summary': summary
        }
    
    def _generate_summary(self, results: List[Dict], filters: Dict) -> str:
        """Generate summary of query results"""
        if not results:
            return "No fraudulent claims found matching your criteria."
        
        total_claims = len(results)
        
        # Calculate totals
        if results and 'metadata' in results[0]:
            total_amount = sum(r['metadata']['amount'] for r in results)
            avg_fraud_score = sum(r['metadata']['fraud_score'] for r in results) / total_claims
        else:
            total_amount = sum(r.get('claim_amount', 0) for r in results)
            avg_fraud_score = sum(r.get('etl_fraud_score', 0) for r in results) / total_claims
        
        specialty = filters.get('specialty', 'All Specialties')
        
        summary = f"""
📊 QUERY SUMMARY
{'='*60}
Specialty:              {specialty}
Total Flagged Claims:   {total_claims}
Total Amount at Risk:   ${total_amount:,.2f}
Average Fraud Score:    {avg_fraud_score:.1f}/100

⚠️  These claims require immediate attention and review.
"""
        
        return summary


def main():
    """Main execution - demonstrate RAG system"""
    print("="*60)
    print("🤖 INITIALIZING RAG SYSTEM")
    print("="*60)
    
    # Load processed data
    claims_df = pd.read_csv('data/processed/claims_processed.csv')
    
    # Load knowledge graph
    from knowledge_graph import HealthcareFraudKnowledgeGraph
    kg = HealthcareFraudKnowledgeGraph()
    
    if os.path.exists('data/processed/knowledge_graph.json'):
        print("   📂 Loading existing knowledge graph...")
        # For demo, rebuild graph
        providers_df = pd.read_csv('data/raw/providers.csv')
        patients_df = pd.read_csv('data/raw/patients.csv')
        graph = kg.build_graph(claims_df, providers_df, patients_df)
    else:
        graph = None
    
    # Initialize RAG
    rag = FraudDetectionRAG(graph=graph)
    rag.index_claims(claims_df)
    
    print("\n✨ RAG System ready!")
    
    # Example queries
    example_queries = [
        "Show me suspicious cardiology claims last month",
        "Find high-risk oncology claims with abnormal amounts",
        "What are the critical fraud cases in orthopedics?"
    ]
    
    print("\n" + "="*60)
    print("🔍 RUNNING EXAMPLE QUERIES")
    print("="*60)
    
    for query in example_queries[:1]:  # Run first query as demo
        print(f"\n{'─'*60}")
        result = rag.query(query, n_results=3)
        
        print(result['summary'])
        
        if result['reports']:
            print(f"\n📄 DETAILED REPORT FOR TOP CLAIM:")
            print(result['reports'][0])


if __name__ == "__main__":
    main()

