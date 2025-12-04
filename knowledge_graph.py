"""
Knowledge Graph Construction for Healthcare Fraud Detection
Builds relationship graph for multi-hop reasoning
"""

import pandas as pd
import networkx as nx
import json
from typing import Dict, List, Tuple, Set
import os


class HealthcareFraudKnowledgeGraph:
    """Build and query knowledge graph for fraud detection"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_attributes = {}
        self.edge_types = set()
        
    def build_graph(self, claims_df: pd.DataFrame, providers_df: pd.DataFrame, 
                   patients_df: pd.DataFrame) -> nx.MultiDiGraph:
        """Build knowledge graph from processed data"""
        
        print("🕸️  Building Knowledge Graph...")
        
        # Add provider nodes
        print("   👨‍⚕️ Adding provider nodes...")
        self._add_provider_nodes(providers_df)
        
        # Add patient nodes
        print("   👥 Adding patient nodes...")
        self._add_patient_nodes(patients_df)
        
        # Add claim nodes and relationships
        print("   📋 Adding claim nodes and relationships...")
        self._add_claim_nodes(claims_df)
        
        # Add medical code nodes
        print("   🩺 Adding medical code nodes...")
        self._add_medical_code_nodes(claims_df)
        
        # Create provider networks
        print("   🔗 Creating provider network relationships...")
        self._create_provider_networks(claims_df, providers_df)
        
        # Add fraud pattern nodes
        print("   🚨 Adding fraud pattern relationships...")
        self._add_fraud_patterns(claims_df)
        
        print(f"\n   ✅ Graph built successfully!")
        print(f"      Nodes: {self.graph.number_of_nodes()}")
        print(f"      Edges: {self.graph.number_of_edges()}")
        print(f"      Edge Types: {len(self.edge_types)}")
        
        return self.graph
    
    def _add_provider_nodes(self, providers_df: pd.DataFrame):
        """Add provider nodes to graph"""
        for _, provider in providers_df.iterrows():
            node_id = f"PROVIDER_{provider['provider_id']}"
            
            self.graph.add_node(
                node_id,
                node_type='Provider',
                provider_id=provider['provider_id'],
                name=provider['name'],
                specialty=provider['specialty'],
                license_number=provider['license_number'],
                years_experience=provider['years_experience'],
                fraud_history_count=provider['fraud_history_count'],
                fraud_types=provider['fraud_types'],
                risk_score=provider['risk_score']
            )
    
    def _add_patient_nodes(self, patients_df: pd.DataFrame):
        """Add patient nodes to graph"""
        for _, patient in patients_df.iterrows():
            node_id = f"PATIENT_{patient['patient_id']}"
            
            self.graph.add_node(
                node_id,
                node_type='Patient',
                patient_id=patient['patient_id'],
                age=patient['age'],
                gender=patient['gender'],
                state=patient['state'],
                member_since=patient['member_since']
            )
    
    def _add_claim_nodes(self, claims_df: pd.DataFrame):
        """Add claim nodes and create relationships"""
        for _, claim in claims_df.iterrows():
            claim_node = f"CLAIM_{claim['claim_id']}"
            provider_node = f"PROVIDER_{claim['provider_id']}"
            patient_node = f"PATIENT_{claim['patient_id']}"
            
            # Add claim node
            self.graph.add_node(
                claim_node,
                node_type='Claim',
                claim_id=claim['claim_id'],
                amount=claim['claim_amount'],
                date=claim['claim_date'],
                is_fraudulent=claim.get('etl_is_fraudulent', 0),
                fraud_score=claim.get('etl_fraud_score', 0),
                fraud_flags=claim.get('etl_fraud_flags', '')
            )
            
            # Create relationships
            # Claim -> Provider (BILLED_BY)
            self.graph.add_edge(
                claim_node, provider_node,
                relationship='BILLED_BY',
                specialty=claim['specialty']
            )
            self.edge_types.add('BILLED_BY')
            
            # Claim -> Patient (FOR_PATIENT)
            self.graph.add_edge(
                claim_node, patient_node,
                relationship='FOR_PATIENT'
            )
            self.edge_types.add('FOR_PATIENT')
    
    def _add_medical_code_nodes(self, claims_df: pd.DataFrame):
        """Add procedure and diagnosis nodes"""
        # Get unique procedures and diagnoses
        procedures = claims_df[['cpt_code', 'procedure_name', 'specialty']].drop_duplicates()
        diagnoses = claims_df[['icd_code', 'diagnosis_name']].drop_duplicates()
        
        # Add procedure nodes
        for _, proc in procedures.iterrows():
            proc_node = f"PROCEDURE_{proc['cpt_code']}"
            
            if not self.graph.has_node(proc_node):
                self.graph.add_node(
                    proc_node,
                    node_type='Procedure',
                    cpt_code=proc['cpt_code'],
                    name=proc['procedure_name'],
                    specialty=proc['specialty']
                )
        
        # Add diagnosis nodes
        for _, diag in diagnoses.iterrows():
            diag_node = f"DIAGNOSIS_{diag['icd_code']}"
            
            if not self.graph.has_node(diag_node):
                self.graph.add_node(
                    diag_node,
                    node_type='Diagnosis',
                    icd_code=diag['icd_code'],
                    name=diag['diagnosis_name']
                )
        
        # Link claims to procedures and diagnoses
        for _, claim in claims_df.iterrows():
            claim_node = f"CLAIM_{claim['claim_id']}"
            proc_node = f"PROCEDURE_{claim['cpt_code']}"
            diag_node = f"DIAGNOSIS_{claim['icd_code']}"
            
            # Claim -> Procedure (HAS_PROCEDURE)
            self.graph.add_edge(
                claim_node, proc_node,
                relationship='HAS_PROCEDURE'
            )
            self.edge_types.add('HAS_PROCEDURE')
            
            # Claim -> Diagnosis (HAS_DIAGNOSIS)
            self.graph.add_edge(
                claim_node, diag_node,
                relationship='HAS_DIAGNOSIS'
            )
            self.edge_types.add('HAS_DIAGNOSIS')
    
    def _create_provider_networks(self, claims_df: pd.DataFrame, providers_df: pd.DataFrame):
        """Create provider network relationships (shared patients, same specialty)"""
        # Find providers with shared patients
        patient_providers = claims_df.groupby('patient_id')['provider_id'].apply(list).to_dict()
        
        shared_patient_pairs = {}
        for patient_id, provider_list in patient_providers.items():
            if len(provider_list) > 1:
                unique_providers = list(set(provider_list))
                for i, p1 in enumerate(unique_providers):
                    for p2 in unique_providers[i+1:]:
                        pair = tuple(sorted([p1, p2]))
                        shared_patient_pairs[pair] = shared_patient_pairs.get(pair, 0) + 1
        
        # Add SHARES_PATIENTS edges
        for (p1, p2), count in shared_patient_pairs.items():
            if count >= 2:  # At least 2 shared patients
                self.graph.add_edge(
                    f"PROVIDER_{p1}",
                    f"PROVIDER_{p2}",
                    relationship='SHARES_PATIENTS',
                    shared_count=count
                )
                self.edge_types.add('SHARES_PATIENTS')
        
        # Add SAME_SPECIALTY edges
        specialty_groups = providers_df.groupby('specialty')['provider_id'].apply(list).to_dict()
        
        for specialty, provider_list in specialty_groups.items():
            if len(provider_list) > 1:
                for i, p1 in enumerate(provider_list):
                    for p2 in provider_list[i+1:]:
                        self.graph.add_edge(
                            f"PROVIDER_{p1}",
                            f"PROVIDER_{p2}",
                            relationship='SAME_SPECIALTY',
                            specialty=specialty
                        )
                        self.edge_types.add('SAME_SPECIALTY')
    
    def _add_fraud_patterns(self, claims_df: pd.DataFrame):
        """Add fraud pattern nodes and relationships"""
        fraudulent_claims = claims_df[claims_df['etl_is_fraudulent'] == 1]
        
        for _, claim in fraudulent_claims.iterrows():
            claim_node = f"CLAIM_{claim['claim_id']}"
            
            # Parse fraud flags
            if pd.notna(claim.get('etl_fraud_flags')) and claim['etl_fraud_flags']:
                flags = claim['etl_fraud_flags'].split(',')
                
                for flag in flags:
                    if flag:
                        fraud_pattern_node = f"FRAUD_PATTERN_{flag.upper()}"
                        
                        # Add fraud pattern node if not exists
                        if not self.graph.has_node(fraud_pattern_node):
                            self.graph.add_node(
                                fraud_pattern_node,
                                node_type='FraudPattern',
                                pattern_type=flag
                            )
                        
                        # Link claim to fraud pattern
                        self.graph.add_edge(
                            claim_node, fraud_pattern_node,
                            relationship='HAS_FRAUD_PATTERN',
                            severity='high' if claim['etl_fraud_score'] > 70 else 'medium'
                        )
                        self.edge_types.add('HAS_FRAUD_PATTERN')
    
    def query_provider_network(self, provider_id: str) -> Dict:
        """Query provider's network and fraud associations"""
        provider_node = f"PROVIDER_{provider_id}"
        
        if not self.graph.has_node(provider_node):
            return {}
        
        # Get provider attributes
        provider_data = dict(self.graph.nodes[provider_node])
        
        # Find connected providers
        connected_providers = []
        for neighbor in self.graph.neighbors(provider_node):
            if self.graph.nodes[neighbor]['node_type'] == 'Provider':
                edge_data = list(self.graph[provider_node][neighbor].values())[0]
                connected_providers.append({
                    'provider_id': self.graph.nodes[neighbor]['provider_id'],
                    'name': self.graph.nodes[neighbor]['name'],
                    'relationship': edge_data['relationship'],
                    'risk_score': self.graph.nodes[neighbor]['risk_score']
                })
        
        # Find associated claims
        claims = []
        for source, target, edge_data in self.graph.in_edges(provider_node, data=True):
            if self.graph.nodes[source]['node_type'] == 'Claim':
                claims.append({
                    'claim_id': self.graph.nodes[source]['claim_id'],
                    'amount': self.graph.nodes[source]['amount'],
                    'fraud_score': self.graph.nodes[source]['fraud_score'],
                    'is_fraudulent': self.graph.nodes[source]['is_fraudulent']
                })
        
        return {
            'provider': provider_data,
            'connected_providers': connected_providers,
            'claims': claims,
            'total_claims': len(claims),
            'fraudulent_claims': sum(1 for c in claims if c['is_fraudulent'] == 1)
        }
    
    def query_fraud_network(self, fraud_pattern: str) -> Dict:
        """Query all claims and providers associated with a fraud pattern"""
        pattern_node = f"FRAUD_PATTERN_{fraud_pattern.upper()}"
        
        if not self.graph.has_node(pattern_node):
            return {}
        
        # Find all claims with this pattern
        related_claims = []
        related_providers = set()
        
        for source, target, edge_data in self.graph.in_edges(pattern_node, data=True):
            if self.graph.nodes[source]['node_type'] == 'Claim':
                claim_data = dict(self.graph.nodes[source])
                
                # Find provider for this claim
                for _, provider_node, _ in self.graph.out_edges(source, data=True):
                    if self.graph.nodes[provider_node]['node_type'] == 'Provider':
                        provider_data = self.graph.nodes[provider_node]
                        related_providers.add(provider_data['provider_id'])
                        claim_data['provider_name'] = provider_data['name']
                        claim_data['provider_specialty'] = provider_data['specialty']
                
                related_claims.append(claim_data)
        
        return {
            'fraud_pattern': fraud_pattern,
            'total_claims': len(related_claims),
            'total_amount': sum(c['amount'] for c in related_claims),
            'providers_involved': len(related_providers),
            'claims': related_claims
        }
    
    def multi_hop_query(self, start_specialty: str, min_fraud_score: int = 70) -> List[Dict]:
        """
        Multi-hop query: Find high-risk fraud patterns
        Specialty -> Claims -> Fraud Patterns -> Connected Providers
        """
        results = []
        
        # Find all high-scoring fraudulent claims in specialty
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data['node_type'] == 'Claim' and 
                node_data['fraud_score'] >= min_fraud_score):
                
                # Get provider
                provider_info = None
                for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                    if (self.graph.nodes[target]['node_type'] == 'Provider' and
                        self.graph.nodes[target]['specialty'] == start_specialty):
                        provider_info = dict(self.graph.nodes[target])
                        break
                
                if provider_info:
                    # Get fraud patterns
                    fraud_patterns = []
                    for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                        if self.graph.nodes[target]['node_type'] == 'FraudPattern':
                            fraud_patterns.append(self.graph.nodes[target]['pattern_type'])
                    
                    # Get patient
                    patient_info = None
                    for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                        if self.graph.nodes[target]['node_type'] == 'Patient':
                            patient_info = dict(self.graph.nodes[target])
                            break
                    
                    results.append({
                        'claim': dict(node_data),
                        'provider': provider_info,
                        'patient': patient_info,
                        'fraud_patterns': fraud_patterns
                    })
        
        return results
    
    def save_graph(self, output_path: str = 'data/processed/knowledge_graph.json'):
        """Save graph to JSON format"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        graph_data = {
            'nodes': [
                {'id': node_id, **data}
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        print(f"   ✅ Saved knowledge graph to {output_path}")


def main():
    """Main execution"""
    print("="*60)
    print("🕸️  BUILDING KNOWLEDGE GRAPH")
    print("="*60)
    
    # Load processed data
    claims_df = pd.read_csv('data/processed/claims_processed.csv')
    providers_df = pd.read_csv('data/raw/providers.csv')
    patients_df = pd.read_csv('data/raw/patients.csv')
    
    # Build knowledge graph
    kg = HealthcareFraudKnowledgeGraph()
    graph = kg.build_graph(claims_df, providers_df, patients_df)
    
    # Save graph
    kg.save_graph()
    
    print("\n✨ Knowledge Graph construction complete!")
    
    # Example query
    print("\n" + "="*60)
    print("🔍 EXAMPLE QUERY: Cardiology High-Risk Claims")
    print("="*60)
    
    results = kg.multi_hop_query('Cardiology', min_fraud_score=70)
    print(f"Found {len(results)} high-risk cardiology claims")
    
    if results:
        print("\nTop 3 High-Risk Claims:")
        for i, result in enumerate(results[:3], 1):
            claim = result['claim']
            provider = result['provider']
            print(f"\n{i}. Claim {claim['claim_id']}")
            print(f"   Provider: {provider['name']} (Risk: {provider['risk_score']})")
            print(f"   Amount: ${claim['amount']:,.2f}")
            print(f"   Fraud Score: {claim['fraud_score']}")
            print(f"   Patterns: {', '.join(result['fraud_patterns'])}")


if __name__ == "__main__":
    main()

