# kobert import.
from kobert_transformers import get_tokenizer

# transformers import.
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

#for pad_sequences and device check
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pytorch import.
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# others import.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_epochs = 2
warmup_ratio = 0
max_grad_norm = 1.0
learning_rate = 2e-5
epsilon = 1e-8
log_interval = 50

# nsmc 데이터 로드
dataset = load_dataset('nsmc')

# 데이터셋 구조 확인
print(dataset)

# 필요한 데이터인 document와 label 정보만 pandas라이브러리 DataFrame 형식으로 변환
train_data = pd.DataFrame({"document":dataset['train']['document'], "label":dataset['train']['label'],})
test_data = pd.DataFrame({"document":dataset['test']['document'], "label":dataset['test']['label'],})

# 데이터셋 갯수 확인
print('학습 데이터셋 : {}'.format(len(train_data)))
print('테스트 데이터셋 : {}'.format(len(test_data)))

# 데이터셋 내용 확인
print(train_data[:5])
print(test_data[:5])

# 데이터 중복을 제외한 갯수 확인
print("학습데이터 : ",train_data['document'].nunique()," 라밸 : ",train_data['label'].nunique())
print("데스트 데이터 : ",test_data['document'].nunique()," 라벨 : ",test_data['label'].nunique())

# 중복 데이터 제거
train_data.drop_duplicates(subset=['document'], inplace= True)
test_data.drop_duplicates(subset=['document'], inplace= True)

# 데이터셋 갯수 확인
print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
print('중복 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))

# null 데이터 제거
train_data['document'].replace('', np.nan, inplace=True)
test_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')

print('null 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
print('null 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))

# BERT 입력 형식에 맞도록 [CLS], [SEP] 추가
train_sentences = train_data['document']
test_sentences = test_data['document']
train_sentences = ["[CLS] " + str(s) + " [SEP]" for s in train_sentences]
test_sentences = ["[CLS] " + str(s) + " [SEP]" for s in test_sentences]
print(train_sentences[:10])
print(test_sentences[:10])

#정답 라벨 추출
y_train = train_data['label'].values
y_test = test_data['label'].values
print(y_train,'\t', y_test)

# 문장을 WordPiece방식으로 tokenizing
tokenizer = get_tokenizer()
train_tokenized = [tokenizer.tokenize(s) for s in train_sentences]
test_tokenized = [tokenizer.tokenize(s) for s in test_sentences]
print(train_sentences[0])
print(test_tokenized[0])

# 토큰을 숫자 인덱스로 변환
x_train = [tokenizer.convert_tokens_to_ids(s) for s in train_tokenized]
x_test = [tokenizer.convert_tokens_to_ids(s) for s in test_tokenized]
print(x_train[0])
print(x_test[0])

#학습 리뷰 길이조사
print('학습 리뷰의 최대 길이 :',max(len(l) for l in x_train))
print('리뷰의 평균 길이 :',sum(map(len, x_train))/len(x_train))

plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of data')
plt.ylabel('number of data')
plt.show()

# 패딩 추가
MAX_LEN = 128# 리뷰 길이 분포를 고려하여 입력 토큰의 최대 시퀀스를 제한

# MAX_LEN 길이로 문장을 자르고 나머지 부분을 0으로 채움
x_train = pad_sequences(x_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print(x_train[0])
print(x_test[0])

# 어텐션 마스크 설정을 위한 함수
def attention_masks(data):
  atten_mask = []
  for seq in data:
      seq_mask = [float(i>0) for i in seq]# 패딩 부분에 0을 넣어 어텐션을 수행하지 않도록 함
      atten_mask.append(seq_mask)
  return atten_mask

# 데이터셋을 학습셋과 검증셋으로 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(x_train, y_train, test_size=0.1, random_state=777)
train_masks, validation_masks, _, _ = train_test_split(attention_masks(x_train), x_train, test_size=0.1, random_state=777)

# 데이터를 파이토치 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)

validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

test_inputs = torch.tensor(x_test)
test_labels = torch.tensor(y_test)
test_masks = torch.tensor(attention_masks(x_test))

# train, validation, test에 대한 DataLoader 구성
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# pretrained kobert model 불러오기
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2).to(device)# 긍정,부정에 대한 이진분류

# 학습 optimizer, scheduler 정의
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon) # adamW 사용.

t_total = len(train_dataloader) * num_epochs # 학습될 전체 데이터 갯수.
warmup_step = int(t_total * warmup_ratio) # warmup step 횟수 설정.

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

# acc 계산 함수
def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    train_acc = np.sum(pred_flat == labels_flat) / len(labels_flat)
    return train_acc

# 재현을 위해 랜덤시드 고정
seed_val = 777
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

''' 학습 '''

for epoch_i in range(num_epochs):  # 설정한 num_epochs만큼 반복
    print('\n', '======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
    total_loss = 0.0
    model.train()  # 훈련 모드

    for step, batch in enumerate(train_dataloader):  # 데이터로더에서 배치만큼 반복하여 가져옴
        batch = tuple(t.to(device) for t in batch)  # 설정한 디바이스에 배치 할당
        b_input_ids, b_input_mask, b_labels = batch  # 배치에서 데이터 추출

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)  # Forward 수행

        loss = outputs[0]  # loss 값
        total_loss += loss.item()  # total_loss에 합산
        loss.backward()  # backward 수행
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 그래디언트 클리핑
        optimizer.step()  # 그래디언트를 통해 가중치 파라미터 업데이트
        scheduler.step()  # 스케줄러로 학습률 업데이트
        model.zero_grad()  # 그래디언트 초기화

        # 경과 정보 표시
        if step % log_interval == 0 and not step == 0:
            print('batch id {} / {}.\t loss {}.'.format(step, len(train_dataloader), loss.data.cpu().numpy()))

    avg_train_loss = total_loss / len(train_dataloader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    model.eval()  # 평가 모드
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:  # 데이터로더에서 배치만큼 반복하여 가져옴
        batch = tuple(t.to(device) for t in batch)  # 설정한 디바이스에 배치 할당
        b_input_ids, b_input_mask, b_labels = batch  # 배치에서 데이터 추출

        with torch.no_grad():  # 그래디언트 계산 비활성화
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  # Forward 수행

        logits = outputs[0]  # loss 값

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력된 logits과 label_ids을 비교하여 정확도 계산
        tmp_eval_accuracy = calc_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print("\n", "Training complete!")


'''테스트 데이터 평가'''

model.eval()  # 평가모드로 변경
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

for step, batch in enumerate(test_dataloader):  # 데이터로더에서 배치만큼 반복하여 가져옴
    # 경과 정보 표시
    if step % 100 == 0 and not step == 0:
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    batch = tuple(t.to(device) for t in batch)  # 설정한 디바이스에 배치 할당

    b_input_ids, b_input_mask, b_labels = batch  # 배치에서 데이터 추출

    with torch.no_grad():  # 그래디언트 계산 비활성화
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]  # loss 값

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # 출력 로짓과 라벨을 비교하여 정확도 계산
    tmp_eval_accuracy = calc_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

    # 경과 정보 표시
    if step % log_interval == 0 and not step == 0:
        print('batch id {} / {}.\t loss {}.'.format(step, len(test_dataloader), logits.data.cpu().numpy()))

print("\n", "Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))


''' 새로운 문장 예측'''


# 입력 문장 전처리 함수
def input_preprocessing(sentences):
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenized]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating='post', padding='post')

    inputs = torch.tensor(input_ids)
    atten_masks = torch.tensor(attention_masks(input_ids))
    return inputs, atten_masks

# predict함수
def sentences_predict(sentences):
    sentences = [sentences]
    model.eval()
    inputs, atten_masks = input_preprocessing(sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = atten_masks.to(device)

    with torch.no_grad():  # 그라디엔트 계산 비활성화
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0].detach().cpu().numpy()
    result = np.argmax(logits)
    return result

sentences_predict("영화 핵꿀 잼ㅋㅋㅋ")
sentences_predict("영화 너무 핵노잼ㅠㅠ")