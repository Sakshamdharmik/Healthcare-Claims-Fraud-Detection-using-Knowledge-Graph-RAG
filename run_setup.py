"""
Complete Setup Script for Healthcare Fraud Detection System
Run this script to set up everything from scratch
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False


def main():
    """Main setup process"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Healthcare Fraud Detection System - Complete Setup          ║
    ║  Built for Abacus Insights Hackathon                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        ("python data_generator.py", "Generating Synthetic Data (1000 claims)"),
        ("python etl_pipeline.py", "Running ETL Pipeline with Fraud Detection"),
        ("python knowledge_graph.py", "Building Knowledge Graph"),
    ]
    
    # Execute all steps
    success = True
    for cmd, description in steps:
        if not run_command(cmd, description):
            success = False
            break
    
    if success:
        print(f"\n{'='*60}")
        print("✨ SETUP COMPLETE! ✨")
        print(f"{'='*60}")
        print("\n📊 System Status:")
        print("   ✅ Data generated: 1000 claims")
        print("   ✅ ETL pipeline executed")
        print("   ✅ Knowledge graph built")
        print("   ✅ Ready for demo!")
        
        print("\n🚀 Next Steps:")
        print("\n   Launch the Streamlit app:")
        print("   >>> streamlit run app.py")
        
        print("\n   Or test the RAG system:")
        print("   >>> python rag_system.py")
        
        print("\n" + "="*60)
        print("Ready for Abacus Insights Hackathon Demo! 🏆")
        print("="*60 + "\n")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

