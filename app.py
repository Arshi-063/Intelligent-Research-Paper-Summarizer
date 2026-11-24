import io
import time
import pandas as pd
import streamlit as st
from src.core import FIELDS, extract_pdf, chunk_pages, Retriever, build_prompt, generate_answer, extractive_fallback, check_answer
from src.generation import endpoint_answer
st.set_page_config(page_title='Research Paper Intelligence V3',layout='wide')
st.title('Research Paper Intelligence | V3')
st.caption('Evidence-grounded research summarization and comparison')
with st.sidebar:
    mode=st.selectbox('Generation mode',['Local Hugging Face LLM','Evidence only (fastest)','Compatible API endpoint'])
    k=st.slider('Evidence chunks per question',2,8,6)
    model_id=st.text_input('Hugging Face model','Qwen/Qwen2.5-1.5B-Instruct') if mode.startswith('Local') else ''
    if mode.startswith('Compatible'):
        base=st.text_input('API base URL','http://localhost:8000/v1'); remote=st.text_input('Server model ID'); key=st.text_input('API key',type='password')
    st.caption('NeMo customization and TensorRT-LLM require separate deployment. API compatibility is not proof of either.')
@st.cache_resource(show_spinner='Loading local model...')
def load_llm(name):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map='auto',trust_remote_code=False).eval()
    return model,tok
uploads=st.file_uploader('Upload research PDFs',type='pdf',accept_multiple_files=True)
if st.button('Analyze papers',type='primary',disabled=not uploads):
    rows=[]; timings=[]; tables=[]
    st.session_state.pop('analysis',None)
    for upload in uploads:
        start=time.perf_counter(); paper_rows=[]; paper_tables=[]
        try:
            with st.status(f'Analyzing {upload.name}') as status:
                pages=extract_pdf(io.BytesIO(upload.getvalue())); chunks=chunk_pages(pages)
                if not chunks: raise ValueError('No selectable text; OCR required.')
                for p in pages:
                    if p['warning']: st.warning(p['warning'])
                    for i,t in enumerate(p['tables']): paper_tables.append((upload.name,p['page'],i+1,t))
                retriever=Retriever(chunks)
                for field,query in FIELDS.items():
                    st.write(field); evidence=retriever.search(query,k,field); prompt=build_prompt(field,evidence)
                    if mode.startswith('Local'): answer=generate_answer(prompt,*load_llm(model_id))
                    elif mode.startswith('Compatible'): answer=endpoint_answer(prompt,base,remote,key)
                    else: answer=extractive_fallback(evidence)
                    issues=check_answer(answer,evidence) if not mode.startswith('Evidence') else []
                    paper_rows.append(dict(paper=upload.name,field=field,answer=answer,retrieved_pages=', '.join(map(str,sorted({e['page'] for e in evidence}))),mode=mode,review='; '.join(issues) or ('Extracted evidence' if mode.startswith('Evidence') else 'Basic checks passed; manual review required'),evidence=evidence))
                status.update(label=f'Completed {upload.name}',state='complete')
            rows.extend(paper_rows); tables.extend(paper_tables); timings.append(dict(paper=upload.name,seconds=round(time.perf_counter()-start,2),mode=mode))
        except Exception as exc: st.error(f'{upload.name}: {type(exc).__name__}: {exc}')
    if rows: st.session_state['analysis']=(rows,timings,tables)
if 'analysis' in st.session_state:
    rows,timings,tables=st.session_state['analysis']; df=pd.DataFrame(rows).drop(columns='evidence'); runtime=pd.DataFrame(timings)
    a,b,c=st.columns(3); a.metric('Papers',df.paper.nunique()); b.metric('Sections',len(df)); c.metric('Total seconds',round(runtime.seconds.sum(),2))
    summary,comparison,metrics,evaluation=st.tabs(['Summaries','Comparison & export','Source tables','Evaluation'])
    with summary:
        for paper in df.paper.unique():
            st.subheader(paper)
            for r in rows:
                if r['paper']!=paper: continue
                with st.expander(r['field']):
                    st.markdown(r['answer']); st.caption(r['review']); st.caption('Retrieved pages: '+r['retrieved_pages'])
                    if st.checkbox('Show evidence',key=f"{paper}-{r['field']}"):
                        for e in r['evidence']: st.write(f"Page {e['page']} | {e['section']} | hybrid score {e['score']:.3f}"); st.text(e['text'])
    with comparison:
        st.dataframe(df,use_container_width=True); st.download_button('Download CSV',df.to_csv(index=False).encode('utf-8-sig'),'paper_comparison.csv')
        report='\n\n'.join(f"{r['paper']} | {r['field']}\n{r['answer']}\nRetrieved pages: {r['retrieved_pages']}\nReview: {r['review']}" for r in rows)
        st.download_button('Download TXT',report,'research_summary.txt')
        import html
        st.download_button('Download printable HTML', '<meta charset="utf-8"><title>Research summary</title><h1>Research summary</h1><pre style="white-space:pre-wrap;font:16px sans-serif">'+html.escape(report)+'</pre>','research_summary.html','text/html')
    with metrics:
        st.caption('Automatically detected tables; verify row and column alignment against the PDF. Detection may miss borderless tables.')
        if not tables: st.info('No tables detected automatically. Inspect results evidence.')
        for paper,page,i,t in tables:
            st.write(f'{paper} | Page {page} | Table {i}'); st.dataframe(pd.DataFrame(t),use_container_width=True)
        if tables:
            import json
            st.download_button('Download detected tables JSON',json.dumps(tables,ensure_ascii=False,indent=2),'source_tables.json')
    with evaluation:
        st.dataframe(runtime); st.bar_chart(runtime.set_index('paper')['seconds']); st.caption('Elapsed time includes retrieval and model loading where applicable. Not a controlled GPU speed benchmark.')
        st.dataframe(df[['paper','field','review']]); st.caption('Citation and number checks are heuristic, not factual accuracy scores.')
        idx=st.selectbox('Select summary to evaluate',range(len(rows)),format_func=lambda i:rows[i]['paper']+' | '+rows[i]['field'])
        ref=st.text_area('Paste a manually verified reference summary')
        if ref.strip():
            from rouge_score import rouge_scorer
            scores=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True).score(ref,rows[idx]['answer'])
            st.dataframe(pd.DataFrame([dict(metric=k,precision=v.precision,recall=v.recall,f1=v.fmeasure) for k,v in scores.items()])); st.caption('ROUGE measures lexical overlap, not factual correctness.')
