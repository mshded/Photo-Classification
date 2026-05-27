import pandas as pd

from src.metrics import compute_classification_metrics
from src.parser import extract_img_candidates_from_html, deduplicate_candidates
from src.features import build_ml_feature_frame, is_probable_tracking_pixel
from src.classifier import _assign_group_splits


def test_metrics_smoke():
    m = compute_classification_metrics([1,1,0,0],[1,0,1,0])
    assert m['tp']==1 and m['fp']==1 and m['fn']==1 and m['tn']==1
    assert abs(m['precision']-0.5)<1e-9


def test_parser_smoke():
    html='''<html><body><img src="/a.jpg" alt="a"><img data-src="/b.jpg" srcset="/c.jpg 1x, /d.jpg 2x"><picture><source srcset="/e.jpg 1x"></picture></body></html>'''
    c=extract_img_candidates_from_html(html,'https://example.com/p')
    u={x['image_url'] for x in deduplicate_candidates(c)}
    assert 'https://example.com/a.jpg' in u
    assert 'https://example.com/b.jpg' in u
    assert 'https://example.com/c.jpg' in u
    assert 'https://example.com/e.jpg' in u


def test_features_nan_and_tracking():
    df=pd.DataFrame([{'image_url':None,'file_name':float('nan'),'alt_text':None,'domain':None,'source_attr':None,'width':1,'height':1,'area':1,'aspect_ratio':1.0,'file_size_bytes':100,'format':'png'}])
    f=build_ml_feature_frame(df)
    assert len(f)==1
    assert is_probable_tracking_pixel(1,1,100,'https://x/track/pixel','x')


def test_split_group_reproducible():
    df=pd.DataFrame({'target':[1,0,1,0,1,0],'page_stub':['p1','p1','p2','p2','p3','p3']})
    a=_assign_group_splits(df,random_state=42)
    b=_assign_group_splits(df,random_state=42)
    assert a['split'].tolist()==b['split'].tolist()
    for s1 in ['train','val','test']:
        for s2 in ['train','val','test']:
            if s1<s2:
                g1=set(a.loc[a['split']==s1,'page_stub'])
                g2=set(a.loc[a['split']==s2,'page_stub'])
                assert g1.isdisjoint(g2)
