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


def test_duplicate_urls():
    df = pd.DataFrame({
        "target": [1, 1, 0, 0, 1, 0],
        "image_url": [
            "https://site.org/image.jpg?w=100",
            "https://site.org/image.jpg?w=500",
            "https://site.org/logo.png",
            "https://site.org/banner.png",
            "https://site.org/photo2.jpg",
            "https://site.org/icon.png",
        ],
    })

    split_df = _assign_group_splits(df, random_state=42)

    duplicate_rows = split_df.iloc[[0, 1]]
    assert duplicate_rows["split"].nunique() == 1