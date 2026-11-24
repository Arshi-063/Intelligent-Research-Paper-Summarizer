from src.core import chunk_pages, build_prompt

def test_chunk_page_provenance():
    chunks=chunk_pages([{'page':7,'text':'example '*400}],chunk_chars=200,overlap=20)
    assert chunks and all(x['page']==7 for x in chunks)

def test_prompt_citations():
    prompt=build_prompt('Dataset',[{'page':3,'text':'Dataset has 100 records.'}])
    assert '[Page 3]' in prompt and 'Do not invent' in prompt

from src.core import BM25, check_answer

def test_bm25():
    assert BM25(['dataset sample size','future work']).score('dataset')[0]>0

def test_citation_check():
    e=[{'page':3,'text':'100 records'}]
    assert not check_answer('100 records [Page 3]',e)
    assert check_answer('200 records [Page 3]',e)
    assert check_answer('100 records [Page 4]',e)

def test_sections():
    c=chunk_pages([{'page':1,'text':'Abstract\nStudy aim.\nReferences\nOther paper.'}])
    assert all('Other paper' not in x['text'] for x in c)
