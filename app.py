
from datetime import datetime, timedelta
from flask import Flask, abort, request
from gensim import corpora, matutils
from gensim.models import TfidfModel
from janome.tokenizer import Tokenizer
import json

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

import numpy as np
import random
import requests
import os

app = Flask(__name__)

line_token = os.environ.get("TOKEN")
line_secret = os.environ.get("SECRET")
notion_token = os.environ.get("NOTION_TOKEN")
notion_database_id = os.environ.get("NOTION_DATABASE_ID")
gpt_seerver_url = os.environ.get("GPT_SERVER_URL")

line_bot_api = LineBotApi(line_token)
handler = WebhookHandler(line_secret)

@app.route("/callback", methods=("GET", "POST"))
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    jp_time = datetime.now() + timedelta(hours=9)

    notion_headers = {"Authorization": f"Bearer {notion_token}",
            "Content-Type": "application/json","Notion-Version": "2021-05-13"}
    notion_body = {"parent": { "database_id": notion_database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": event.message.text}}]},
            "Created": {"date": {"start": jp_time.isoformat()}}
        }}
    requests.request('POST', url='https://api.notion.com/v1/pages',\
        headers=notion_headers, data=json.dumps(notion_body))

    line_bot_api.reply_message(
        event.reply_token,
        [TextSendMessage(text=make_reply(event.message.text))]
    )

def make_reply(text):
    if text.lower().startswith("gpt"):
        print(text.lower())
        return chat_reply(text[3:])
    else:
        return cos_meigen(text)

def chat_reply(text):
    gpt_headers = {
        "Content-Type": "application/json",
    }
    gpt_data = f'{{"content":"{text}"}}'

    gpt_response = requests.post(gpt_seerver_url, headers=gpt_headers, data=gpt_data.encode("utf-8"))
    gpt_result = gpt_response.json()
    return gpt_result["reply"]

def token_generator(text, stopwords):
    tokens = []
    t = Tokenizer()
    for token in t.tokenize(text):
        if token.surface not in stopwords:
            if (token.part_of_speech.split(',')[0] == '名詞')\
                or (token.part_of_speech.split(',')[0] == '形容詞')\
                    or (token.part_of_speech.split(',')[0] == '動詞')\
                        or (token.part_of_speech.split(',')[0] == '副詞'):
                tokens.append(token.surface)
    return tokens

def cos_meigen(src):
    meigen = np.load("meigen.pkl", allow_pickle=True)
    text_processed = np.load("tokens.pkl", allow_pickle=True)
    dictionary = corpora.Dictionary(text_processed)
    stop_words = []

    if "返信不要" in src:
        return ""

    src = [src]
    outcome = meigen + src
    targettoken = [token_generator(src[0], stop_words)]
    text_processed3 = text_processed + targettoken
    corpus = [dictionary.doc2bow(doc) for doc in text_processed3]
    model = TfidfModel(corpus)
    tfidf = [model[i] for i in corpus]
    doc_matrix = matutils.corpus2csc(tfidf).transpose()
    c = len(outcome)
    cos_sim = np.zeros([c, c])
    var_SDGs = doc_matrix.dot(doc_matrix.transpose()).toarray()

    for j in range(c):
        cos_sim[-1, j] = var_SDGs[-1, j]/(np.sqrt(var_SDGs[-1, -1])*np.sqrt(var_SDGs[j, j]))

    simirality_for_the_target3 = cos_sim[c-1:c, :-1]
    A2_result = np.argsort(-simirality_for_the_target3)
    if A2_result[0, 0] != 0:
        return meigen[A2_result[0, 0]]
    else:
        return "全然違うこと言うけど、" + meigen[random.randint(0, len(meigen)-1)]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
