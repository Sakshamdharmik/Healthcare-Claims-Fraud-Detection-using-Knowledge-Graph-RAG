"""
Healthcare Claims Fraud Detection - Synthetic Data Generator
Generates realistic healthcare claims data with injected fraud patterns
for Abacus Insights Hackathon
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class HealthcareDataGenerator:
    """Generate synthetic healthcare claims data with fraud patterns"""
    
    def __init__(self, n_claims: int = 1000):
        self.n_claims = n_claims
        self.n_providers = 50
        self.n_patients = 300
        
        # Medical coding data
        self.specialties = [
            'Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 
            'Dermatology', 'Primary Care', 'Emergency Medicine'
        ]
        
        self.procedures = {
            'Cardiology': [
                ('93458', 'Cardiac catheterization', 8000, 12000),
                ('93015', 'Stress test', 500, 1500),
                ('93306', 'Echocardiogram', 800, 1800),
                ('92928', 'Angioplasty', 15000, 25000),
                ('93000', 'ECG', 100, 300)
            ],
            'Oncology': [
                ('96413', 'Chemotherapy administration', 3000, 8000),
                ('77427', 'Radiation therapy', 5000, 12000),
                ('38220', 'Bone marrow biopsy', 2000, 4000),
                ('99215', 'Office visit complex', 200, 400),
                ('36415', 'Blood draw', 50, 100)
            ],
            'Orthopedics': [
                ('27447', 'Total knee replacement', 20000, 35000),
                ('29881', 'Knee arthroscopy', 5000, 10000),
                ('99213', 'Office visit moderate', 150, 300),
                ('73721', 'MRI lower extremity', 1200, 2500),
                ('20610', 'Joint injection', 300, 600)
            ],
            'Neurology': [
                ('95860', 'EMG testing', 800, 1500),
                ('70553', 'MRI brain with contrast', 2000, 4000),
                ('99214', 'Office visit detailed', 180, 350),
                ('95886', 'Nerve conduction study', 600, 1200),
                ('64483', 'Lumbar epidural injection', 1500, 3000)
            ],
            'Dermatology': [
                ('17000', 'Destroy lesion', 200, 500),
                ('11100', 'Skin biopsy', 300, 600),
                ('99212', 'Office visit simple', 100, 200),
                ('17110', 'Destroy warts', 150, 350),
                ('11042', 'Debridement', 250, 550)
            ],
            'Primary Care': [
                ('99213', 'Office visit moderate', 120, 250),
                ('99214', 'Office visit detailed', 160, 320),
                ('36415', 'Venipuncture', 30, 80),
                ('85025', 'Complete blood count', 50, 120),
                ('99395', 'Preventive visit adult', 150, 300)
            ],
            'Emergency Medicine': [
                ('99285', 'Emergency visit high severity', 800, 2000),
                ('99283', 'Emergency visit moderate', 400, 800),
                ('71020', 'Chest X-ray', 150, 350),
                ('36556', 'IV insertion', 100, 250),
                ('12001', 'Simple wound repair', 300, 700)
            ]
        }
        
        self.diagnoses = {
            'Cardiology': [
                ('I20.0', 'Unstable angina'),
                ('I21.9', 'Acute myocardial infarction'),
                ('I50.9', 'Heart failure'),
                ('I48.91', 'Atrial fibrillation'),
                ('I25.10', 'Coronary artery disease')
            ],
            'Oncology': [
                ('C50.9', 'Breast cancer'),
                ('C61', 'Prostate cancer'),
                ('C34.90', 'Lung cancer'),
                ('C18.9', 'Colon cancer'),
                ('C92.10', 'Chronic myeloid leukemia')
            ],
            'Orthopedics': [
                ('M17.0', 'Bilateral knee osteoarthritis'),
                ('M25.561', 'Pain in right knee'),
                ('S83.201A', 'Meniscus tear'),
                ('M79.661', 'Pain in lower leg'),
                ('M23.90', 'Knee disorder')
            ],
            'Neurology': [
                ('G43.909', 'Migraine'),
                ('G40.909', 'Epilepsy'),
                ('G35', 'Multiple sclerosis'),
                ('G20', "Parkinson's disease"),
                ('G47.00', 'Insomnia')
            ],
            'Dermatology': [
                ('L70.0', 'Acne vulgaris'),
                ('L40.9', 'Psoriasis'),
                ('C44.90', 'Skin cancer'),
                ('L30.9', 'Dermatitis'),
                ('B07.9', 'Viral warts')
            ],
            'Primary Care': [
                ('E11.9', 'Type 2 diabetes'),
                ('I10', 'Essential hypertension'),
                ('J06.9', 'Upper respiratory infection'),
                ('E78.5', 'Hyperlipidemia'),
                ('Z00.00', 'General health checkup')
            ],
            'Emergency Medicine': [
                ('S06.0X0A', 'Concussion'),
                ('J18.9', 'Pneumonia'),
                ('S52.501A', 'Fracture of forearm'),
                ('R07.9', 'Chest pain'),
                ('K35.80', 'Acute appendicitis')
            ]
        }
        
        # Fraud patterns configuration
        self.fraud_rate = 0.15  # 15% of claims will have fraud indicators
        
    def generate_providers(self) -> pd.DataFrame:
        """Generate provider data"""
        providers = []
        
        for i in range(self.n_providers):
            provider_id = f"P{i+1:04d}"
            specialty = random.choice(self.specialties)
            
            # Some providers will have fraud history
            has_fraud_history = random.random() < 0.20  # 20% of providers
            
            if has_fraud_history:
                fraud_count = random.randint(1, 8)
                fraud_types = random.sample(
                    ['duplicate_billing', 'upcoding', 'unbundling', 'diagnosis_mismatch'],
                    k=min(random.randint(1, 3), 4)
                )
            else:
                fraud_count = 0
                fraud_types = []
            
            providers.append({
                'provider_id': provider_id,
                'name': f"Dr. {self._generate_name()}",
                'specialty': specialty,
                'license_number': f"{random.choice(['CA', 'NY', 'TX', 'FL'])}{random.randint(10000, 99999)}",
                'years_experience': random.randint(3, 35),
                'fraud_history_count': fraud_count,
                'fraud_types': ','.join(fraud_types) if fraud_types else None,
                'risk_score': fraud_count * 15 if has_fraud_history else random.randint(0, 20)
            })
        
        return pd.DataFrame(providers)
    
    def generate_patients(self) -> pd.DataFrame:
        """Generate patient data"""
        patients = []
        
        for i in range(self.n_patients):
            patient_id = f"PT{i+1:05d}"
            
            patients.append({
                'patient_id': patient_id,
                'age': random.randint(18, 85),
                'gender': random.choice(['M', 'F']),
                'state': random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA']),
                'member_since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(patients)
    
    def generate_claims(self, providers_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Generate claims data with fraud patterns"""
        claims = []
        
        # Track for duplicate billing fraud
        duplicate_candidates = []
        
        start_date = datetime.now() - timedelta(days=180)  # Last 6 months
        
        for i in range(self.n_claims):
            claim_id = f"CLM{i+1:06d}"
            
            # Select provider and patient
            provider = providers_df.sample(1).iloc[0]
            patient = patients_df.sample(1).iloc[0]
            
            specialty = provider['specialty']
            
            # Select procedure and diagnosis from specialty
            procedure_info = random.choice(self.procedures[specialty])
            diagnosis_info = random.choice(self.diagnoses[specialty])
            
            cpt_code, procedure_name, min_cost, max_cost = procedure_info
            icd_code, diagnosis_name = diagnosis_info
            
            # Base claim amount
            claim_amount = round(random.uniform(min_cost, max_cost), 2)
            
            # Claim date
            claim_date = start_date + timedelta(days=random.randint(0, 180))
            
            # Initialize fraud flags
            fraud_flags = []
            is_fraudulent = False
            
            # FRAUD PATTERN 1: Duplicate Billing (5% of claims)
            if random.random() < 0.05 and len(duplicate_candidates) > 0:
                # Create duplicate of existing claim
                duplicate = random.choice(duplicate_candidates)
                provider = duplicate['provider']
                patient = duplicate['patient']
                cpt_code = duplicate['cpt_code']
                procedure_name = duplicate['procedure_name']
                icd_code = duplicate['icd_code']
                diagnosis_name = duplicate['diagnosis_name']
                claim_amount = duplicate['claim_amount']
                claim_date = duplicate['claim_date'] + timedelta(hours=random.randint(1, 48))
                
                fraud_flags.append('duplicate_billing')
                is_fraudulent = True
            
            # FRAUD PATTERN 2: Procedure-Diagnosis Mismatch (4% of claims)
            elif random.random() < 0.04:
                # Use procedure from one specialty but diagnosis from another
                wrong_specialty = random.choice([s for s in self.specialties if s != specialty])
                icd_code, diagnosis_name = random.choice(self.diagnoses[wrong_specialty])
                
                fraud_flags.append('diagnosis_mismatch')
                is_fraudulent = True
            
            # FRAUD PATTERN 3: Abnormal Amount / Upcoding (4% of claims)
            elif random.random() < 0.04:
                # Inflate claim amount significantly
                claim_amount = round(claim_amount * random.uniform(2.5, 5.0), 2)
                fraud_flags.append('abnormal_amount')
                is_fraudulent = True
            
            # FRAUD PATTERN 4: High-risk provider pattern (2% of claims)
            elif provider['fraud_history_count'] > 3 and random.random() < 0.02:
                # Providers with fraud history more likely to have suspicious patterns
                claim_amount = round(claim_amount * random.uniform(1.8, 3.0), 2)
                fraud_flags.append('high_risk_provider')
                is_fraudulent = True
            
            # Add to duplicate candidates pool (keep last 100)
            if not is_fraudulent and cpt_code not in ['99212', '99213', '36415']:  # Not routine procedures
                duplicate_candidates.append({
                    'provider': provider,
                    'patient': patient,
                    'cpt_code': cpt_code,
                    'procedure_name': procedure_name,
                    'icd_code': icd_code,
                    'diagnosis_name': diagnosis_name,
                    'claim_amount': claim_amount,
                    'claim_date': claim_date
                })
                if len(duplicate_candidates) > 100:
                    duplicate_candidates.pop(0)
            
            claim = {
                'claim_id': claim_id,
                'provider_id': provider['provider_id'],
                'provider_name': provider['name'],
                'specialty': specialty,
                'patient_id': patient['patient_id'],
                'cpt_code': cpt_code,
                'procedure_name': procedure_name,
                'icd_code': icd_code,
                'diagnosis_name': diagnosis_name,
                'claim_amount': claim_amount,
                'claim_date': claim_date.strftime('%Y-%m-%d'),
                'claim_datetime': claim_date.strftime('%Y-%m-%d %H:%M:%S'),
                'is_fraudulent': is_fraudulent,
                'fraud_flags': ','.join(fraud_flags) if fraud_flags else None,
                'fraud_score': self._calculate_fraud_score(fraud_flags, provider, claim_amount, max_cost)
            }
            
            claims.append(claim)
        
        return pd.DataFrame(claims)
    
    def _calculate_fraud_score(self, fraud_flags: List[str], provider: pd.Series, 
                               claim_amount: float, expected_max: float) -> int:
        """Calculate fraud risk score (0-100)"""
        score = 0
        
        if 'duplicate_billing' in fraud_flags:
            score += 40
        if 'diagnosis_mismatch' in fraud_flags:
            score += 35
        if 'abnormal_amount' in fraud_flags:
            score += 30
        if 'high_risk_provider' in fraud_flags:
            score += 20
        
        # Add provider risk
        score += min(provider['risk_score'], 25)
        
        # Add amount deviation risk
        if claim_amount > expected_max * 1.5:
            score += 15
        
        return min(score, 100)
    
    def _generate_name(self) -> str:
        """Generate random provider name"""
        first_names = ['James', 'Mary', 'John', 'Jennifer', 'Michael', 'Linda', 'David', 
                      'Susan', 'Robert', 'Jessica', 'William', 'Sarah', 'Richard', 'Karen',
                      'Joseph', 'Nancy', 'Thomas', 'Lisa', 'Charles', 'Betty']
        
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                     'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                     'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets"""
        print("🏥 Generating Healthcare Claims Data for Fraud Detection...")
        print(f"📊 Target: {self.n_claims} claims with ~{int(self.n_claims * self.fraud_rate)} fraudulent patterns\n")
        
        print("👨‍⚕️ Generating providers...")
        providers_df = self.generate_providers()
        print(f"   ✅ Created {len(providers_df)} providers across {len(self.specialties)} specialties")
        
        print("👥 Generating patients...")
        patients_df = self.generate_patients()
        print(f"   ✅ Created {len(patients_df)} patients")
        
        print("📋 Generating claims with fraud patterns...")
        claims_df = self.generate_claims(providers_df, patients_df)
        
        fraud_count = claims_df['is_fraudulent'].sum()
        fraud_pct = (fraud_count / len(claims_df)) * 100
        
        print(f"   ✅ Created {len(claims_df)} claims")
        print(f"   🚨 Fraudulent claims: {fraud_count} ({fraud_pct:.1f}%)")
        
        # Print fraud pattern breakdown
        print("\n📊 Fraud Pattern Distribution:")
        for flag_type in ['duplicate_billing', 'diagnosis_mismatch', 'abnormal_amount', 'high_risk_provider']:
            count = claims_df['fraud_flags'].str.contains(flag_type, na=False).sum()
            print(f"   • {flag_type.replace('_', ' ').title()}: {count} claims")
        
        return {
            'providers': providers_df,
            'patients': patients_df,
            'claims': claims_df
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = 'data/raw'):
        """Save generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving data to {output_dir}/...")
        
        for name, df in data.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"   ✅ Saved {filepath} ({len(df)} rows)")
        
        print("\n✨ Data generation complete!")


def main():
    """Main execution"""
    generator = HealthcareDataGenerator(n_claims=1000)
    data = generator.generate_all_data()
    generator.save_data(data)
    
    # Print sample statistics
    claims_df = data['claims']
    
    print("\n" + "="*60)
    print("📈 DATASET STATISTICS")
    print("="*60)
    print(f"Total Claims: {len(claims_df)}")
    print(f"Date Range: {claims_df['claim_date'].min()} to {claims_df['claim_date'].max()}")
    print(f"Total Claim Amount: ${claims_df['claim_amount'].sum():,.2f}")
    print(f"Average Claim: ${claims_df['claim_amount'].mean():,.2f}")
    print(f"Fraudulent Claims Amount: ${claims_df[claims_df['is_fraudulent']]['claim_amount'].sum():,.2f}")
    
    print("\n🏥 Claims by Specialty:")
    print(claims_df['specialty'].value_counts().to_string())
    
    print("\n⚠️ High-Risk Claims (Fraud Score > 70):")
    high_risk = claims_df[claims_df['fraud_score'] > 70]
    print(f"Count: {len(high_risk)}")
    if len(high_risk) > 0:
        print(f"Total Amount: ${high_risk['claim_amount'].sum():,.2f}")


if __name__ == "__main__":
    main()

