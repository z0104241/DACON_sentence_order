#!/usr/bin/env python3
"""
가벼운 후처리 스크립트
간단하고 빠른 시간 표현 분석
"""

import pandas as pd
import re
import argparse
from typing import List, Dict, Tuple, Union
import time
from collections import Counter

def extract_first_words_and_patterns(sentence: str) -> List[str]:
    """문장에서 첫 단어들과 시간 패턴 추출"""
    
    # 전처리: 불필요한 문자 제거
    clean_sentence = re.sub(r'[^\w\s가-힣]', ' ', sentence)
    words = clean_sentence.split()
    
    if not words:
        return []
    
    # 첫 2-3개 단어 추출 (시간 표현은 보통 문장 앞부분에)
    first_words = words[:3]
    
    # 추가로 전체 문장에서 시간 표현 패턴 검색
    temporal_patterns = [
        r'먼저', r'처음', r'우선', r'일단', r'시작',
        r'그다음', r'이후', r'다음', r'그러고\s*나서', r'계속',
        r'그리고', r'또한', r'한편', r'동시에',
        r'그래서', r'따라서', r'결국', r'마지막', r'최종'
    ]
    
    found_patterns = []
    for pattern in temporal_patterns:
        if re.search(pattern, sentence):
            found_patterns.append(pattern.replace(r'\s*', ''))
    
    return first_words + found_patterns

def analyze_temporal_order_simple(sentences: List[str]) -> List[int]:
    """간단한 시간 순서 분석"""
    
    # 시간 표현별 우선순위 (더 포괄적)
    time_priorities = {
        # 강한 시작 표현 (-3점)
        '먼저': -3, '처음': -3, '우선': -3, '일단': -3, '시작': -3,
        '가장먼저': -3, '제일먼저': -3, '우선적으로': -3,
        
        # 초기 단계 (-2점)  
        '그다음': -2, '이후': -2, '다음': -2, '그후': -2, '그뒤': -2,
        '계속': -2, '이어서': -2, '그러고나서': -2,
        
        # 중간 단계 (-1점)
        '그리고': -1, '또한': -1, '한편': -1, '동시에': -1,
        '그런데': -1, '그러나': -1,
        
        # 중립 (0점)
        '그': 0, '이': 0, '그녀': 0, '그들': 0,
        
        # 후기 단계 (+1점)
        '그래서': 1, '따라서': 1, '그결과': 1, '이에': 1,
        '그러므로': 1, '그로인해': 1,
        
        # 강한 마지막 표현 (+2점)
        '마지막': 2, '결국': 2, '최종적으로': 2, '끝으로': 2,
        '드디어': 2, '마침내': 2, '최종': 2, '결론적으로': 2,
        
        # 매우 강한 마지막 (+3점)
        '가장마지막': 3, '제일마지막': 3, '최종적': 3
    }
    
    sentence_scores = []
    
    for i, sentence in enumerate(sentences):
        total_score = 0
        found_expressions = extract_first_words_and_patterns(sentence)
        
        # 각 표현의 점수 합산
        for expr in found_expressions:
            clean_expr = re.sub(r'\s+', '', expr.lower())
            if clean_expr in time_priorities:
                total_score += time_priorities[clean_expr]
        
        # 문장 길이도 약간 고려 (긴 문장은 보통 중간이나 마지막)
        length_bonus = min(len(sentence) // 20, 1) * 0.1
        total_score += length_bonus
        
        sentence_scores.append((i, total_score))
    
    # 점수순으로 정렬
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1])
    
    # 순위 할당 (가장 낮은 점수가 1순위)
    priorities = [0] * 4
    for rank, (sent_idx, score) in enumerate(sorted_sentences):
        priorities[sent_idx] = rank + 1
    
    return priorities

def enhanced_pattern_matching(sentences: List[str]) -> str:
    """강화된 패턴 매칭 기반 후처리"""
    
    # 매우 명확한 순서 지시어들
    explicit_order_patterns = {
        r'첫\s*번째': 1, r'1번째': 1, r'첫째': 1,
        r'두\s*번째': 2, r'2번째': 2, r'둘째': 2,  
        r'세\s*번째': 3, r'3번째': 3, r'셋째': 3,
        r'네\s*번째': 4, r'4번째': 4, r'넷째': 4
    }
    
    sentence_positions = [-1] * 4
    
    # 명시적 순서 지시어 찾기
    for i, sentence in enumerate(sentences):
        for pattern, position in explicit_order_patterns.items():
            if re.search(pattern, sentence):
                sentence_positions[position-1] = i
                break
    
    # 명시적 지시어가 2개 이상 있으면 그것을 기준으로
    explicit_count = sum(1 for pos in sentence_positions if pos != -1)
    if explicit_count >= 2:
        # 빈 자리는 원래 순서 유지
        result = []
        remaining = [i for i in range(4) if i not in sentence_positions]
        remaining_idx = 0
        
        for pos in sentence_positions:
            if pos != -1:
                result.append(str(pos))
            else:
                result.append(str(remaining[remaining_idx]))
                remaining_idx += 1
        
        return ''.join(result)
    
    return None  # 명시적 지시어가 부족하면 다른 방법 사용

def question_answer_pattern(sentences: List[str]) -> str:
    """질문-답변 패턴 인식"""
    
    question_patterns = [r'\?', r'까\?', r'나\?', r'는가\?', r'일까\?', r'무엇', r'어떻게', r'왜', r'언제']
    answer_patterns = [r'답:', r'답은', r'대답', r'~다\.', r'~이다\.', r'~한다\.']
    
    question_sentences = []
    answer_sentences = []
    
    for i, sentence in enumerate(sentences):
        # 질문 패턴 체크
        for pattern in question_patterns:
            if re.search(pattern, sentence):
                question_sentences.append(i)
                break
        
        # 답변 패턴 체크  
        for pattern in answer_patterns:
            if re.search(pattern, sentence):
                answer_sentences.append(i)
                break
    
    # 질문이 있고 답변이 있으면 질문을 답변 앞으로
    if len(question_sentences) == 1 and len(answer_sentences) >= 1:
        q_idx = question_sentences[0]
        a_idx = answer_sentences[0]
        
        # 간단한 재배열: 질문 → 답변 순서로
        remaining = [i for i in range(4) if i not in [q_idx, a_idx]]
        return f"{q_idx}{a_idx}{remaining[0]}{remaining[1]}"
    
    return None

def lightweight_postprocessing(predicted_order: str, sentences: List[str]) -> str:
    """가벼운 후처리 메인 함수"""
    
    # 1순위: 명시적 순서 지시어
    explicit_result = enhanced_pattern_matching(sentences)
    if explicit_result:
        return explicit_result
    
    # 2순위: 질문-답변 패턴
    qa_result = question_answer_pattern(sentences)
    if qa_result:
        return qa_result
    
    # 3순위: 시간 표현 분석
    try:
        priorities = analyze_temporal_order_simple(sentences)
        
        # 우선순위가 명확히 다른 경우만 적용
        unique_priorities = len(set(priorities))
        if unique_priorities >= 3:
            # 우선순위를 순서로 변환
            order_pairs = [(i, priorities[i]) for i in range(4)]
            sorted_pairs = sorted(order_pairs, key=lambda x: x[1])
            new_order = ''.join(str(pair[0]) for pair in sorted_pairs)
            return new_order
    except:
        pass
    
    # 모든 방법이 실패하면 원본 유지
    return predicted_order

def apply_lightweight_postprocessing(input_file: str, output_file: str) -> None:
    """가벼운 후처리 적용"""
    
    print(f"📂 AI 추론 결과 로드: {input_file}")
    df = pd.read_csv(input_file)
    
    postprocessed_results = []
    changes_count = 0
    method_stats = Counter()
    
    print("⚡ 가벼운 후처리 적용 중...")
    
    for idx, row in df.iterrows():
        if row['parsing_status'] == 'SUCCESS':
            original_order = f"{row['answer_0']}{row['answer_1']}{row['answer_2']}{row['answer_3']}"
            sentences = [row['sentence_0'], row['sentence_1'], row['sentence_2'], row['sentence_3']]
            
            # 각 방법별로 시도해서 어떤 방법이 효과적인지 추적
            explicit_result = enhanced_pattern_matching(sentences)
            qa_result = question_answer_pattern(sentences)
            
            # 최종 후처리 적용
            postprocessed_order = lightweight_postprocessing(original_order, sentences)
            
            # 변경 원인 추적
            if postprocessed_order != original_order:
                changes_count += 1
                if explicit_result and explicit_result == postprocessed_order:
                    method_stats['explicit_order'] += 1
                elif qa_result and qa_result == postprocessed_order:
                    method_stats['question_answer'] += 1
                else:
                    method_stats['temporal_analysis'] += 1
                
                print(f"🔄 변경 {row['ID']}: {original_order} → {postprocessed_order}")
            
            # 결과 저장
            postprocessed_results.append({
                'ID': row['ID'],
                'answer_0': int(postprocessed_order[0]),
                'answer_1': int(postprocessed_order[1]),
                'answer_2': int(postprocessed_order[2]),
                'answer_3': int(postprocessed_order[3]),
                'raw_output': '',
                'parsing_status': 'SUCCESS'
            })
        else:
            # 파싱 실패 케이스는 그대로 유지
            postprocessed_results.append({
                'ID': row['ID'],
                'answer_0': row['answer_0'],
                'answer_1': row['answer_1'],
                'answer_2': row['answer_2'],
                'answer_3': row['answer_3'],
                'raw_output': row['raw_output'],
                'parsing_status': row['parsing_status']
            })
    
    # 결과 저장
    postprocessed_df = pd.DataFrame(postprocessed_results)
    postprocessed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 통계 출력
    print(f"💾 가벼운 후처리 결과 저장: {output_file}")
    print(f"🔄 총 변경된 케이스: {changes_count}개")
    print(f"📊 변경 비율: {(changes_count/len(df)*100):.1f}%")
    
    print(f"\n📈 방법별 효과:")
    for method, count in method_stats.items():
        print(f"   {method}: {count}개")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='가벼운 후처리 (빠르고 효율적)')
    parser.add_argument('--input', '-i', default='ai_predictions.csv', help='AI 추론 결과 CSV 파일')
    parser.add_argument('--output', '-o', default='lightweight_postprocessed.csv', help='후처리 결과 CSV 파일')
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        
        # 가벼운 후처리 적용
        apply_lightweight_postprocessing(args.input, args.output)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  처리 시간: {elapsed_time:.1f}초 (매우 빠름!)")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()