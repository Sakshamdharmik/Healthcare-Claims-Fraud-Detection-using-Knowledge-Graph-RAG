"""
ETL Pipeline with ML-Based Fraud Detection
Uses trained machine learning model for predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import pickle


class FraudDetectionETL_ML:
    """ETL Pipeline with ML-based fraud detection"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        self.claims_df = None
        self.providers_df = None
        self.patients_df = None
        self.model = None
        self.model_loaded = False
        
    def load_data(self):
        """Load raw data from CSV files"""
        print("📂 Loading data...")
        
        self.claims_df = pd.read_csv(os.path.join(self.data_dir, 'claims.csv'))
        self.providers_df = pd.read_csv(os.path.join(self.data_dir, 'providers.csv'))
        self.patients_df = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
        
        print(f"   ✅ Loaded {len(self.claims_df)} claims")
        print(f"   ✅ Loaded {len(self.providers_df)} providers")
        print(f"   ✅ Loaded {len(self.patients_df)} patients")
        
    def load_ml_model(self, model_path: str = 'models/fraud_detection_model.pkl'):
        """Load trained ML model"""
        print("\n🤖 Loading ML model...")
        
        if not os.path.exists(model_path):
            print(f"   ❌ Model not found at: {model_path}")
            print("   Please train the model first: python ml_model_trainer.py")
            return False
        
        try:
            from ml_model_trainer_fixed import FraudDetectionMLModel
            
            self.model = FraudDetectionMLModel()
            self.model.load_model()
            self.model_loaded = True
            
            print("   ✅ ML model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return False
    
    def apply_ml_predictions(self) -> pd.DataFrame:
        """Apply ML model predictions"""
        
        if not self.model_loaded:
            print("❌ ML model not loaded! Using rule-based fallback...")
            return self.apply_fraud_rules_fallback()
        
        print("\n🤖 Applying ML-based fraud detection...")
        
        claims = self.claims_df.copy()
        
        try:
            # Get ML predictions
            predictions, probabilities = self.model.predict(
                claims, self.providers_df, self.patients_df
            )
            
            # Add ML predictions to claims
            claims['ml_prediction'] = predictions
            claims['ml_fraud_probability'] = probabilities
            claims['ml_fraud_score'] = (probabilities * 100).round(0).astype(int)
            
            # Determine if fraudulent (using probability threshold)
            threshold = 0.5
            claims['etl_is_fraudulent'] = (probabilities >= threshold).astype(int)
            claims['etl_fraud_score'] = claims['ml_fraud_score']
            
            # Add explanation flags based on feature importance and probability
            claims['etl_fraud_flags'] = claims.apply(
                lambda row: self._generate_explanation_flags(row, probabilities[row.name]),
                axis=1
            )
            
            fraud_count = claims['etl_is_fraudulent'].sum()
            avg_prob = probabilities.mean()
            
            print(f"   ✅ ML predictions complete")
            print(f"   🚨 Flagged as fraudulent: {fraud_count} ({fraud_count/len(claims)*100:.1f}%)")
            print(f"   📊 Average fraud probability: {avg_prob:.1%}")
            
            return claims
            
        except Exception as e:
            print(f"   ❌ Error in ML predictions: {e}")
            print("   Falling back to rule-based detection...")
            return self.apply_fraud_rules_fallback()
    
    def _generate_explanation_flags(self, row: pd.Series, probability: float) -> str:
        """Generate explanation flags based on claim characteristics"""
        flags = []
        
        # High probability
        if probability > 0.8:
            flags.append('high_confidence_fraud')
        elif probability > 0.6:
            flags.append('likely_fraud')
        
        # High amount
        if row['claim_amount'] > 10000:
            flags.append('high_amount')
        
        # Provider risk
        if 'risk_score' in row and row['risk_score'] > 50:
            flags.append('high_risk_provider')
        
        # Check for potential patterns (simplified)
        if 'fraud_history_count' in row and row['fraud_history_count'] > 0:
            flags.append('provider_fraud_history')
        
        return ','.join(flags) if flags else 'ml_detected'
    
    def apply_fraud_rules_fallback(self) -> pd.DataFrame:
        """Fallback to rule-based detection if ML model unavailable"""
        print("\n🔧 Applying rule-based fraud detection (fallback)...")
        
        claims = self.claims_df.copy()
        
        # Simple rule-based scoring
        claims['etl_fraud_score'] = 0
        claims['etl_fraud_flags'] = ''
        
        # Rule 1: High amount
        amount_threshold = claims['claim_amount'].quantile(0.95)
        high_amount_mask = claims['claim_amount'] > amount_threshold
        claims.loc[high_amount_mask, 'etl_fraud_score'] += 30
        
        # Rule 2: Provider risk (if available)
        if 'fraud_history_count' in self.providers_df.columns:
            claims = claims.merge(
                self.providers_df[['provider_id', 'fraud_history_count', 'risk_score']],
                on='provider_id', how='left', suffixes=('', '_provider')
            )
            high_risk_mask = claims['fraud_history_count'] > 2
            claims.loc[high_risk_mask, 'etl_fraud_score'] += 40
        
        # Determine fraudulent
        claims['etl_is_fraudulent'] = (claims['etl_fraud_score'] > 50).astype(int)
        claims['etl_fraud_flags'] = 'rule_based_detection'
        
        fraud_count = claims['etl_is_fraudulent'].sum()
        print(f"   ✅ Rule-based detection complete")
        print(f"   🚨 Flagged as fraudulent: {fraud_count}")
        
        return claims
    
    def generate_summary_statistics(self, claims: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        stats = {
            'total_claims': len(claims),
            'total_amount': claims['claim_amount'].sum(),
            'fraudulent_claims': claims['etl_is_fraudulent'].sum(),
            'fraudulent_amount': claims[claims['etl_is_fraudulent'] == 1]['claim_amount'].sum(),
            'avg_fraud_score': claims['etl_fraud_score'].mean(),
            'high_risk_claims': len(claims[claims['etl_fraud_score'] > 70]),
            'detection_method': 'ML-based' if self.model_loaded else 'Rule-based',
            'by_specialty': claims.groupby('specialty').agg({
                'claim_id': 'count',
                'claim_amount': 'sum',
                'etl_is_fraudulent': 'sum'
            }).to_dict()
        }
        
        return stats
    
    def save_processed_data(self, claims: pd.DataFrame, output_dir: str = 'data/processed'):
        """Save processed claims data"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving processed data to {output_dir}/...")
        
        # Save all claims
        claims.to_csv(os.path.join(output_dir, 'claims_processed.csv'), index=False)
        print(f"   ✅ Saved claims_processed.csv")
        
        # Save only fraudulent claims
        fraudulent = claims[claims['etl_is_fraudulent'] == 1]
        fraudulent.to_csv(os.path.join(output_dir, 'fraudulent_claims.csv'), index=False)
        print(f"   ✅ Saved fraudulent_claims.csv ({len(fraudulent)} claims)")
        
        # Save high-risk claims
        high_risk = claims[claims['etl_fraud_score'] > 70]
        high_risk.to_csv(os.path.join(output_dir, 'high_risk_claims.csv'), index=False)
        print(f"   ✅ Saved high_risk_claims.csv ({len(high_risk)} claims)")
    
    def run_pipeline(self):
        """Run complete ETL pipeline with ML predictions"""
        print("="*60)
        print("🚀 STARTING ML-BASED ETL PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Load ML model
        self.load_ml_model()
        
        # Apply ML predictions
        processed_claims = self.apply_ml_predictions()
        
        # Generate statistics
        print("\n📊 Generating summary statistics...")
        stats = self.generate_summary_statistics(processed_claims)
        
        print("\n" + "="*60)
        print("📈 FRAUD DETECTION RESULTS")
        print("="*60)
        print(f"Detection Method: {stats['detection_method']}")
        print(f"Total Claims Processed: {stats['total_claims']}")
        print(f"Total Claim Amount: ${stats['total_amount']:,.2f}")
        print(f"\n🚨 Fraudulent Claims: {stats['fraudulent_claims']} ({stats['fraudulent_claims']/stats['total_claims']*100:.1f}%)")
        print(f"💰 Fraudulent Amount: ${stats['fraudulent_amount']:,.2f}")
        print(f"📊 Average Fraud Score: {stats['avg_fraud_score']:.1f}")
        print(f"⚠️  High-Risk Claims (>70): {stats['high_risk_claims']}")
        
        if self.model_loaded:
            print("\n🏥 Fraud by Specialty (ML-detected):")
        else:
            print("\n🏥 Fraud by Specialty (Rule-based):")
            
        for specialty in processed_claims['specialty'].unique():
            specialty_claims = processed_claims[processed_claims['specialty'] == specialty]
            fraud_count = specialty_claims['etl_is_fraudulent'].sum()
            total_count = len(specialty_claims)
            fraud_pct = (fraud_count / total_count * 100) if total_count > 0 else 0
            print(f"   • {specialty}: {fraud_count}/{total_count} ({fraud_pct:.1f}%)")
        
        # Save processed data
        self.save_processed_data(processed_claims)
        
        print("\n✨ ETL Pipeline complete!")
        
        return processed_claims, stats


def main():
    """Main execution"""
    etl = FraudDetectionETL_ML()
    processed_claims, stats = etl.run_pipeline()


if __name__ == "__main__":
    main()

