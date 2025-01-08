import os
import pandas as pd
import numpy as np
import streamlit as st
import openai
from openai import OpenAI
import json
import re

# 設計ルールと列名の統一
RULES = {
    "基板長さ_mm": (50, 200),
    "基板幅_mm": (30, 100),
    "基板厚み_mm": (0.1, 1.0),
    "配線幅_mm": (0.2, 5.0),
    "配線間ギャップ_mm": (0.5, float("inf")),
}

ADDITIONAL_KEYS = ["配線本数", "伸縮率_%", "伸縮回数", "温度_℃"]

def validate_input(inputs):
    """入力値が設計ルールを満たしているか検証"""
    invalid_fields = []
    for key, (lower, upper) in RULES.items():
        val = inputs.get(key, None)
        if val is None or not (lower <= val <= upper):
            invalid_fields.append(key)
    return invalid_fields

def get_sheet_names(length, width):
    """基板サイズに基づいて使用するシート名を返す"""
    if length >= 126 or width >= 51:
        return ["大型品_電気特性", "大型品_機械特性"]
    return ["小型品_電気特性", "小型品_機械特性"]

def calculate_average_error(dfs, user_input):
    """ユーザー入力とデータの誤差を計算"""
    cols = ["基板長さ_mm", "基板幅_mm", "基板厚み_mm", "配線幅_mm", "配線本数", "伸縮率_%", "伸縮回数", "温度_℃"]
    combined_results = pd.DataFrame()
    for sheet_name, df in dfs.items():
        df_filtered = df[cols]
        user_vector = np.array([user_input.get(col, 0) for col in cols])
        df["誤差平均"] = df_filtered.apply(lambda row: np.mean(np.abs(user_vector - row.values)), axis=1)
        df["シート名"] = sheet_name
        combined_results = pd.concat([combined_results, df])
    return combined_results.sort_values(by="誤差平均")

def get_top_similar_data(similar_data, n=5):
    """各シートの上位n件のデータを取得"""
    top_data = {}
    for sheet_name in similar_data['シート名'].unique():
        sheet_data = similar_data[similar_data['シート名'] == sheet_name].head(n)
        top_data[sheet_name] = sheet_data
    return top_data

def prepare_chat_context(top_data):
    """類似データを文字列形式に変換"""
    context = "検索された類似データ:\n\n"
    for sheet_name, data in top_data.items():
        context += f"\n{sheet_name}の上位5件:\n"
        context += data.to_string(index=False)
    return context

# セッション状態の初期化
if "inputs" not in st.session_state:
    st.session_state.inputs = {key: None for key in RULES.keys()}
if "additional_inputs" not in st.session_state:
    st.session_state.additional_inputs = {key: None for key in ADDITIONAL_KEYS}
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("設計値検証アプリと類似データ検索")

# メインの入力フォーム
if not st.session_state.chat_mode:
    st.header("設計値を入力してください")
    for key, (lower, upper) in RULES.items():
        placeholder = f"{lower} ~ {upper} mm"
        value = st.text_input(f"{key}", value="", placeholder=placeholder)
        if value.strip():
            try:
                st.session_state.inputs[key] = float(value)
            except ValueError:
                st.error(f"{key}には数値を入力してください。")
                st.stop()

    invalid_fields = validate_input(st.session_state.inputs)
    if invalid_fields:
        st.error("設計不可能です。以下の項目が不適合です:")
        for field in invalid_fields:
            st.write(f"- {field}")
        st.stop()

    st.success("設計可能です！ 次の項目を入力してください：")
    for col in ADDITIONAL_KEYS:
        value = st.text_input(f"{col}", value="", placeholder="ここに値を入力")
        if value.strip():
            try:
                st.session_state.additional_inputs[col] = float(value)
            except ValueError:
                st.error(f"{col}には数値を入力してください。")
                st.stop()

    full_inputs = {**st.session_state.inputs, **st.session_state.additional_inputs}
    try:
        file_path = "ダミーデータ.xlsx"
        length = st.session_state.inputs["基板長さ_mm"]
        width = st.session_state.inputs["基板幅_mm"]
        sheet_names = get_sheet_names(length, width)
        dfs = {sheet_name: pd.read_excel(file_path, sheet_name=sheet_name) for sheet_name in sheet_names}
        
        similar_data = calculate_average_error(dfs, full_inputs)
        st.write("誤差平均計算結果:")
        st.dataframe(similar_data)

        if not similar_data.empty:
            top_similar = get_top_similar_data(similar_data)
            st.write("### 類似度上位5件")
            for sheet_name, data in top_similar.items():
                st.dataframe(data)
            
            if st.button("OpenAIとチャットする"):
                st.session_state.chat_mode = True
                st.session_state.context_data = prepare_chat_context(top_similar)
                st.session_state.full_inputs_for_chat = full_inputs
                st.rerun()
    except Exception as e:
        st.error(f"データ検索中にエラーが発生しました: {e}")

# チャットモード
else:
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
    if st.button("入力画面に戻る"):
        st.session_state.chat_mode = False
        st.session_state.messages = []
        st.rerun()

    st.write("### チャット履歴")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("質問を入力してください")
    if user_input:
        if not openai_api_key:
            st.info("OpenAI API Keyを入力してください。")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            try:
                client = OpenAI(api_key=openai_api_key)
                system_content = f"""あなたは基板の管理を行なっている会社の従業員として回答してください。ユーザーからの質問に対して与えられたデータや入力された設計値をもとに回答します。

入力された設計値:
{json.dumps(st.session_state.full_inputs_for_chat, indent=2, ensure_ascii=False)}

検索された類似データ:
{st.session_state.context_data}

上記の情報を基に、具体的な数値を示しながら回答してください。
特性値の予想の際には類似データを参考に平均値や分散、回帰的な分析をした上で予測してください。
"""

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input}
                ]

                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")