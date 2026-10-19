"""Content-only manual review, explicitly not preference reannotation."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def main():
    pairs=pd.read_csv(OUT/'content_pairs.csv',dtype={'a_id':str,'b_id':str},keep_default_na=False)
    reviews=json.loads((HERE/'reviews.json').read_text())
    assert set(reviews)==set(pairs.loc[pairs.selection.eq('sample'),'pair_id'])
    rows=[];lines=['# Khảo sát khác biệt nội dung giữa 20 cặp bài train','',
        'Đây là nhận xét về nội dung có thể quan sát, không phải gán lại nhãn sở thích. Nhãn 1/4/5 giữ nguyên. Không biết lý do thật của người gán nhãn. Mẫu được chọn có chủ đích theo similarity và nhãn đối nghịch, không đại diện để ước lượng label noise.','']
    for pair in pairs[pairs.selection.eq('sample')].to_dict('records'):
        review=reviews[pair['pair_id']]
        record={k:pair[k] for k in ['pair_id','venue','a_id','b_id','a_label','b_label','category_flag_differences']}
        record.update(observability='clear_content_difference',preference_rationale='unknown',
                      difference=review['difference'],flags_capture=review['flags_capture'],
                      lexical_artifact=review.get('lexical_artifact',''))
        for role in ['a','b']:
            quote=review[role+'_quote'];text=pair[role+'_abstract']
            assert quote in text,(pair['pair_id'],role,quote)
            start=text.index(quote)
            record.update({role+'_field':'abstract',role+'_start':start,role+'_end':start+len(quote),role+'_evidence':quote})
        rows.append(record)
        lines.extend([f"## {pair['pair_id']} — {pair['venue']} — ID {pair['a_id']} / {pair['b_id']}",'',
            f"**A, nhãn gốc {pair['a_label']}:** {pair['a_title']}",'',
            f"**B, nhãn gốc {pair['b_label']}:** {pair['b_title']}",'',
            review['difference'],'',f"Bằng chứng A: “{review['a_quote']}”",'',
            f"Bằng chứng B: “{review['b_quote']}”",'',
            f"Khác {pair['category_flag_differences']} cờ nội dung. Mức thể hiện khác biệt này trong bộ cờ: {review['flags_capture']}.",'',
            review.get('lexical_artifact',''),'',
            'Các khác biệt này chưa giải thích được vì sao người gán nhãn thích B hơn A.',''])
    pd.DataFrame(rows).to_csv(OUT/'reviewed_pairs.csv',index=False)
    (OUT/'CONTENT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print('Reviewed',len(rows),'pairs; all40 evidence snippets checked against the original usable abstracts.')


if __name__=='__main__':main()
