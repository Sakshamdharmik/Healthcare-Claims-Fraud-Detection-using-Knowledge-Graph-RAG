# 🔧 Troubleshooting Guide

Quick solutions for common issues during setup and demo.

---

## 🚨 Installation Issues

### Error: "ModuleNotFoundError: No module named 'X'"

**Solution:**
```bash
pip install -r requirements.txt
```

**If that doesn't work:**
```bash
# Install packages individually
pip install pandas numpy networkx
pip install chromadb sentence-transformers
pip install streamlit plotly matplotlib seaborn
```

**Still having issues?**
```bash
# Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Error: "pip: command not found"

**Windows:**
```bash
python -m pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m pip install -r requirements.txt
```

---

## 📊 Data Generation Issues

### Error: "FileNotFoundError: data/raw/"

**Problem:** Directory doesn't exist

**Solution:**
```bash
# Create directories manually
mkdir -p data/raw data/processed

# Then run generator
python data_generator.py
```

---

### Error: "PermissionError" when saving files

**Windows Solution:**
- Run terminal as Administrator
- Or check folder permissions

**Mac/Linux Solution:**
```bash
chmod -R 755 data/
python data_generator.py
```

---

## 🔍 ETL Pipeline Issues

### Error: "FileNotFoundError: data/raw/claims.csv"

**Problem:** Data not generated yet

**Solution:**
```bash
# Generate data first
python data_generator.py

# Then run ETL
python etl_pipeline.py
```

---

### Warning: "No fraud patterns detected"

**This is OK!** Some random seeds may generate fewer fraud patterns. The system still works.

**To regenerate with different patterns:**
```bash
python data_generator.py  # Will generate new random data
python etl_pipeline.py    # Re-process
```

---

## 🕸️ Knowledge Graph Issues

### Error: "ModuleNotFoundError: No module named 'networkx'"

**Solution:**
```bash
pip install networkx
```

---

### Issue: Graph building is slow

**Normal!** Building 1,423 nodes and 4,784 edges takes 10-30 seconds depending on your computer.

**Progress indicators:**
- You should see: "Adding provider nodes..." etc.
- Wait for "Graph built successfully!"

---

## 🤖 RAG System Issues

### Warning: "ChromaDB not installed"

**This is OK!** The system falls back to pandas-based search.

**To enable full functionality:**
```bash
pip install chromadb
```

**Note:** ChromaDB may have issues on some systems. The fallback works fine for demo.

---

### Warning: "OpenAI API key not found"

**This is OK!** OpenAI integration is optional. The system works without it.

**To enable (optional):**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## 🖥️ Streamlit Issues

### Error: "streamlit: command not found"

**Solution:**
```bash
# Install streamlit
pip install streamlit

# Or run directly
python -m streamlit run app.py
```

---

### Issue: Streamlit won't open in browser

**Solution 1:** Open manually
- Copy the URL from terminal (usually http://localhost:8501)
- Paste in browser

**Solution 2:** Try different port
```bash
streamlit run app.py --server.port 8502
```

---

### Issue: "Address already in use" error

**Problem:** Port 8501 is busy

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing Streamlit
# Windows: Ctrl+C in terminal, then restart
# Mac/Linux: killall streamlit
```

---

### Issue: App shows "No data found"

**Problem:** Data files don't exist or are empty

**Solution:**
```bash
# Check if files exist
ls data/processed/

# If empty, regenerate
python data_generator.py
python etl_pipeline.py
python knowledge_graph.py

# Then refresh Streamlit (Ctrl+R in browser)
```

---

### Issue: Chatbot queries return no results

**Possible causes:**

1. **Data not loaded**
   ```bash
   # Verify files exist
   ls -lh data/processed/claims_processed.csv
   
   # Should be ~500KB-1MB
   # If not, regenerate data
   ```

2. **ChromaDB initialization failed**
   - Check terminal for warnings
   - System falls back to pandas search (still works)

3. **Query too specific**
   - Try simpler query: "Show me suspicious claims"
   - Avoid: "Show me exactly 3 cardiology claims from Dr. Smith on Tuesday"

---

## 📈 Visualization Issues

### Issue: Charts not rendering

**Problem:** Plotly not installed or conflicting versions

**Solution:**
```bash
pip install --upgrade plotly
pip install kaleido  # For static image export
```

---

### Issue: Network graph visualization is slow

**Normal!** Large graphs take time to render. Be patient (10-20 seconds).

---

## 🚀 Demo Day Issues

### CRITICAL: App crashes during demo

**Emergency Plan:**

1. **Have screenshots ready**
   - Pre-capture dashboard, chatbot results
   - Walk through static images

2. **Use backup terminal demo**
   ```bash
   python rag_system.py
   ```
   This runs a command-line demo without Streamlit

3. **Show the code**
   - Open files in VS Code
   - Walk through architecture
   - "The demo gods were not with us, but the code is solid"

---

### Issue: Query takes too long

**Quick fix:**
- Use the example query buttons (pre-tested, instant)
- Avoid typing complex queries during demo
- Have 3-4 tested queries memorized

---

### Issue: Results look different than expected

**This is normal!** Random seed variations mean each data generation creates slightly different results.

**What to do:**
- Focus on the patterns, not exact numbers
- "In this run, we detected 197 fraudulent claims..."
- Emphasize the methodology, not specific values

---

## 🔒 Security/Privacy Questions

### Q: "Is this HIPAA compliant?"

**Answer:**
"All data is synthetic - zero real patient information. In production, this would run on Abacus's secure platform with:
- Encrypted data at rest and in transit
- Role-based access control
- Complete audit logging
- PHI de-identification pipelines"

---

### Q: "What about data security?"

**Answer:**
"The knowledge graph actually enhances security through:
- Fine-grained access control at node/edge level
- Audit trails for every query
- Data lineage tracking
- Ability to redact sensitive relationships while preserving fraud detection"

---

## 🐛 Common Warnings (Safe to Ignore)

### ✅ These are OK:

```
FutureWarning: ...
  → Pandas/NumPy version warnings, safe to ignore

ChromaDB: anonymized telemetry
  → Privacy feature, can disable in code

Streamlit: Server is running
  → Normal startup message
```

### ⚠️ These need attention:

```
FileNotFoundError: data/
  → Need to generate data

ModuleNotFoundError: 
  → Need to install dependencies

KeyError: 'etl_fraud_score'
  → ETL pipeline didn't run or failed
```

---

## 🔄 Nuclear Option: Complete Reset

If everything is broken:

```bash
# 1. Delete all generated data
rm -rf data/raw/* data/processed/*

# 2. Reinstall dependencies
pip install --upgrade -r requirements.txt

# 3. Run complete setup
python run_setup.py

# 4. Launch app
streamlit run app.py
```

**Windows equivalent:**
```bash
# Delete data
del data\raw\* /Q
del data\processed\* /Q

# Reinstall
pip install --upgrade -r requirements.txt

# Setup and launch
python run_setup.py
streamlit run app.py
```

---

## 📞 Still Stuck?

### Debug Checklist

- [ ] Python 3.9+ installed? (`python --version`)
- [ ] All dependencies installed? (`pip list | grep streamlit`)
- [ ] Data files exist? (`ls data/processed/`)
- [ ] Files not empty? (`ls -lh data/processed/claims_processed.csv`)
- [ ] No syntax errors? (`python -m py_compile app.py`)

### Get System Info

```bash
python --version
pip list
ls -la data/raw/
ls -la data/processed/
```

Send this output if asking for help.

---

## 💡 Pro Tips

### Speed Up Demo Prep

1. **Pre-generate data the night before**
   ```bash
   python run_setup.py
   ```

2. **Test queries beforehand**
   - Write down 3 queries that work well
   - Practice the flow

3. **Have backup plan**
   - Screenshots of key screens
   - Printed summary (PROJECT_SUMMARY.md)
   - Code ready to show

### Make Demo Smoother

1. **Clear browser cache** before demo
2. **Close other apps** (free up memory)
3. **Disable notifications** (focus mode)
4. **Use wired internet** (if possible)
5. **Have charger plugged in** (full power mode)

---

## 🎯 Quick Reference Commands

### Complete Setup (First Time)
```bash
pip install -r requirements.txt
python run_setup.py
streamlit run app.py
```

### Quick Launch (After Setup)
```bash
streamlit run app.py
# Or: launch_app.bat (Windows)
# Or: ./launch_app.sh (Mac/Linux)
```

### Regenerate Data Only
```bash
python data_generator.py
python etl_pipeline.py
python knowledge_graph.py
```

### Test Without UI
```bash
python rag_system.py
```

---

## 🏆 Demo Day Confidence Builder

**Remember:**
- ✅ You have a complete, working system
- ✅ The code is solid (1000+ lines)
- ✅ The concept is innovative (KG RAG)
- ✅ The business case is strong ($60B fraud problem)
- ✅ The Abacus alignment is clear

**Even if something breaks:**
- You can show the code
- You can explain the architecture
- You can discuss the business impact
- You built something impressive

**You've got this! 🚀**

---

*Last updated: December 2025*
*For Abacus Insights Hackathon*

