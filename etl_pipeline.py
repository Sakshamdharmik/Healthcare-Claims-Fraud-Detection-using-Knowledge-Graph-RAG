"""
ETL Pipeline with Fraud Detection Rules
Processes healthcare claims and applies fraud detection logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os


class FraudDetectionETL:
    """ETL Pipeline with comprehensive fraud detection rules"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        self.claims_df = None
        self.providers_df = None
        self.patients_df = None
        
    def load_data(self):
        """Load raw data from CSV files"""
        print("📂 Loading data...")
        
        self.claims_df = pd.read_csv(os.path.join(self.data_dir, 'claims.csv'))
        self.providers_df = pd.read_csv(os.path.join(self.data_dir, 'providers.csv'))
        self.patients_df = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
        
        print(f"   ✅ Loaded {len(self.claims_df)} claims")
        print(f"   ✅ Loaded {len(self.providers_df)} providers")
        print(f"   ✅ Loaded {len(self.patients_df)} patients")
        
    def apply_fraud_rules(self) -> pd.DataFrame:
        """Apply comprehensive fraud detection rules"""
        print("\n🔍 Applying fraud detection rules...")
        
        claims = self.claims_df.copy()
        
        # Rule 1: Duplicate Billing Detection
        print("   📋 Rule 1: Detecting duplicate billing...")
        claims = self._detect_duplicates(claims)
        
        # Rule 2: Abnormal Amount Detection
        print("   💰 Rule 2: Detecting abnormal claim amounts...")
        claims = self._detect_abnormal_amounts(claims)
        
        # Rule 3: Procedure-Diagnosis Mismatch
        print("   🩺 Rule 3: Detecting procedure-diagnosis mismatches...")
        claims = self._detect_medical_mismatches(claims)
        
        # Rule 4: High-Frequency Billing
        print("   📊 Rule 4: Detecting high-frequency billing patterns...")
        claims = self._detect_high_frequency(claims)
        
        # Rule 5: Provider Risk Assessment
        print("   👨‍⚕️ Rule 5: Assessing provider risk...")
        claims = self._assess_provider_risk(claims)
        
        # Rule 6: Temporal Anomalies
        print("   ⏰ Rule 6: Detecting temporal anomalies...")
        claims = self._detect_temporal_anomalies(claims)
        
        # Calculate final fraud scores and flags
        claims = self._calculate_final_scores(claims)
        
        return claims
    
    def _detect_duplicates(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Detect duplicate billing"""
        # Find exact duplicates on key fields within 7 days
        claims['claim_date_dt'] = pd.to_datetime(claims['claim_date'])
        claims = claims.sort_values('claim_date_dt')
        
        # Check for duplicates: same patient, provider, procedure within 3 days
        claims['duplicate_flag'] = 0
        
        for idx, row in claims.iterrows():
            mask = (
                (claims['patient_id'] == row['patient_id']) &
                (claims['provider_id'] == row['provider_id']) &
                (claims['cpt_code'] == row['cpt_code']) &
                (claims['claim_date_dt'] >= row['claim_date_dt']) &
                (claims['claim_date_dt'] <= row['claim_date_dt'] + timedelta(days=3)) &
                (claims.index != idx)
            )
            
            if mask.any():
                claims.loc[idx, 'duplicate_flag'] = 1
        
        dup_count = claims['duplicate_flag'].sum()
        print(f"      🚨 Found {dup_count} potential duplicate claims")
        
        return claims
    
    def _detect_abnormal_amounts(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Detect abnormally high claim amounts"""
        claims['abnormal_amount_flag'] = 0
        
        # Calculate statistics by specialty and procedure
        for specialty in claims['specialty'].unique():
            specialty_mask = claims['specialty'] == specialty
            specialty_claims = claims[specialty_mask]
            
            for cpt_code in specialty_claims['cpt_code'].unique():
                procedure_mask = specialty_mask & (claims['cpt_code'] == cpt_code)
                procedure_claims = claims[procedure_mask]['claim_amount']
                
                if len(procedure_claims) > 5:  # Need enough data
                    mean_amt = procedure_claims.mean()
                    std_amt = procedure_claims.std()
                    
                    # Flag if > 2.5 standard deviations above mean
                    threshold = mean_amt + (2.5 * std_amt)
                    
                    abnormal_mask = procedure_mask & (claims['claim_amount'] > threshold)
                    claims.loc[abnormal_mask, 'abnormal_amount_flag'] = 1
        
        abnormal_count = claims['abnormal_amount_flag'].sum()
        print(f"      🚨 Found {abnormal_count} claims with abnormal amounts")
        
        return claims
    
    def _detect_medical_mismatches(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Detect procedure-diagnosis mismatches"""
        claims['mismatch_flag'] = 0
        
        # Define valid procedure-diagnosis category mappings
        valid_mappings = {
            'Cardiology': ['I', 'R07'],  # Cardiovascular or chest pain codes
            'Oncology': ['C'],  # Cancer codes
            'Orthopedics': ['M', 'S'],  # Musculoskeletal or injury codes
            'Neurology': ['G', 'R51'],  # Neurological codes
            'Dermatology': ['L', 'B07', 'C44'],  # Skin codes
            'Primary Care': ['E', 'I10', 'J', 'Z00'],  # Various primary care codes
            'Emergency Medicine': ['S', 'R', 'J', 'K35']  # Injuries, symptoms, various
        }
        
        for idx, row in claims.iterrows():
            specialty = row['specialty']
            icd_code = row['icd_code']
            
            if specialty in valid_mappings:
                valid_prefixes = valid_mappings[specialty]
                
                # Check if diagnosis code starts with any valid prefix
                is_valid = any(icd_code.startswith(prefix) for prefix in valid_prefixes)
                
                if not is_valid:
                    claims.loc[idx, 'mismatch_flag'] = 1
        
        mismatch_count = claims['mismatch_flag'].sum()
        print(f"      🚨 Found {mismatch_count} procedure-diagnosis mismatches")
        
        return claims
    
    def _detect_high_frequency(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Detect high-frequency billing patterns"""
        claims['high_frequency_flag'] = 0
        
        # Count claims per provider per day
        daily_counts = claims.groupby(['provider_id', 'claim_date']).size().reset_index(name='daily_count')
        
        # Calculate average and flag outliers
        avg_daily = daily_counts['daily_count'].mean()
        std_daily = daily_counts['daily_count'].std()
        threshold = avg_daily + (2 * std_daily)
        
        high_freq_dates = daily_counts[daily_counts['daily_count'] > threshold]
        
        for _, row in high_freq_dates.iterrows():
            mask = (
                (claims['provider_id'] == row['provider_id']) &
                (claims['claim_date'] == row['claim_date'])
            )
            claims.loc[mask, 'high_frequency_flag'] = 1
        
        freq_count = claims['high_frequency_flag'].sum()
        print(f"      🚨 Found {freq_count} claims in high-frequency billing patterns")
        
        return claims
    
    def _assess_provider_risk(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Assess provider risk based on history"""
        # Merge provider fraud history
        claims = claims.merge(
            self.providers_df[['provider_id', 'fraud_history_count', 'risk_score']],
            on='provider_id',
            how='left',
            suffixes=('', '_provider')
        )
        
        # Flag high-risk providers
        claims['high_risk_provider_flag'] = (claims['fraud_history_count'] >= 3).astype(int)
        
        high_risk_count = claims['high_risk_provider_flag'].sum()
        print(f"      🚨 Found {high_risk_count} claims from high-risk providers")
        
        return claims
    
    def _detect_temporal_anomalies(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Detect temporal anomalies (e.g., claims at unusual hours)"""
        claims['temporal_anomaly_flag'] = 0
        
        # Parse datetime if available
        if 'claim_datetime' in claims.columns:
            claims['claim_hour'] = pd.to_datetime(claims['claim_datetime']).dt.hour
            
            # Flag claims submitted at unusual hours (e.g., 1-5 AM)
            unusual_hours_mask = (claims['claim_hour'] >= 1) & (claims['claim_hour'] <= 5)
            claims.loc[unusual_hours_mask, 'temporal_anomaly_flag'] = 1
        
        temporal_count = claims['temporal_anomaly_flag'].sum()
        print(f"      🚨 Found {temporal_count} claims with temporal anomalies")
        
        return claims
    
    def _calculate_final_scores(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Calculate final fraud scores and aggregate flags"""
        # Collect all rule-based flags
        flag_columns = [
            'duplicate_flag',
            'abnormal_amount_flag', 
            'mismatch_flag',
            'high_frequency_flag',
            'high_risk_provider_flag',
            'temporal_anomaly_flag'
        ]
        
        # Calculate weighted fraud score
        weights = {
            'duplicate_flag': 35,
            'abnormal_amount_flag': 25,
            'mismatch_flag': 30,
            'high_frequency_flag': 15,
            'high_risk_provider_flag': 20,
            'temporal_anomaly_flag': 10
        }
        
        claims['etl_fraud_score'] = 0
        for flag, weight in weights.items():
            if flag in claims.columns:
                claims['etl_fraud_score'] += claims[flag] * weight
        
        # Add provider risk score component
        if 'risk_score' in claims.columns:
            claims['etl_fraud_score'] += claims['risk_score'] * 0.3
        
        # Cap at 100
        claims['etl_fraud_score'] = claims['etl_fraud_score'].clip(upper=100).round(0).astype(int)
        
        # Create combined fraud flags string
        claims['etl_fraud_flags'] = claims.apply(
            lambda row: ','.join([
                flag.replace('_flag', '') 
                for flag in flag_columns 
                if flag in row.index and row[flag] == 1
            ]),
            axis=1
        )
        
        # Mark as fraudulent if score > 50 or has critical flags
        critical_flags = ['duplicate_flag', 'mismatch_flag']
        claims['etl_is_fraudulent'] = (
            (claims['etl_fraud_score'] > 50) |
            (claims[critical_flags].sum(axis=1) > 0)
        ).astype(int)
        
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
        """Run complete ETL pipeline"""
        print("="*60)
        print("🚀 STARTING ETL PIPELINE WITH FRAUD DETECTION")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Apply fraud detection rules
        processed_claims = self.apply_fraud_rules()
        
        # Generate statistics
        print("\n📊 Generating summary statistics...")
        stats = self.generate_summary_statistics(processed_claims)
        
        print("\n" + "="*60)
        print("📈 FRAUD DETECTION RESULTS")
        print("="*60)
        print(f"Total Claims Processed: {stats['total_claims']}")
        print(f"Total Claim Amount: ${stats['total_amount']:,.2f}")
        print(f"\n🚨 Fraudulent Claims: {stats['fraudulent_claims']} ({stats['fraudulent_claims']/stats['total_claims']*100:.1f}%)")
        print(f"💰 Fraudulent Amount: ${stats['fraudulent_amount']:,.2f}")
        print(f"📊 Average Fraud Score: {stats['avg_fraud_score']:.1f}")
        print(f"⚠️  High-Risk Claims (>70): {stats['high_risk_claims']}")
        
        print("\n🏥 Fraud by Specialty:")
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
    etl = FraudDetectionETL()
    processed_claims, stats = etl.run_pipeline()


if __name__ == "__main__":
    main()

