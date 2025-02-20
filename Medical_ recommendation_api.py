from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import uvicorn, json, datetime
import torch
import ahocorasick
from py2neo import Graph
import os
import csv
import ast

DEVICE = "cuda"
DEVICE_ID = "2"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
drug_path=os.path.join("/home/username/FindGPT/dict","drug.csv")
test={}
for i in csv.reader(open("/home/username/FindGPT/dict/drug.csv",encoding="utf-8-sig")):
    test[i[0]]=i[1]
drugs=dict()
desc1=[]
resultant_dictionary = dict()
drug_wds= [i[0] for i in csv.reader(open("/home/username/FindGPT/dict/drug.csv",encoding="utf-8-sig")) if i[0]]
qu=False
g = Graph(
        host="10.23.12.109",
        port=7687,
        user="neo4j",
        password="12345678")
def add_medical(response):
    global drugs
    global desc1
    global resultant_dictionary
    disease_path=os.path.join("/home/username/FindGPT/dict","disease.txt")
    disease_wds= [i.strip() for i in open(disease_path,encoding="utf-8") if i.strip()]
    actree = ahocorasick.Automaton()
    for index, word in enumerate(disease_wds):
        actree.add_word(word, (index, word))
    actree.make_automaton()
    region_wds=[]
    final_wds=[]
    sql = []
    answers=[]
    num_limit=10 #限制推荐数量
    for i in actree.iter(response):
        wd = i[1][1]
        region_wds.append(wd)
    if not region_wds:
        return response
    else:
        final_wds=list(set(region_wds))
        print("final_wds:{0}".format(final_wds))

        for final_wds_ in final_wds:
            sql1 = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{0}' return m.name,n.name,id(n)".format(final_wds_)]
            sql2 = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name,n.name,id(n)".format(final_wds_)]
            sql = sql1 + sql2
            for sql_ in sql:
                ress = g.run(sql_).data()
                answers += ress
            # desc = [i['n.name'] for i in answers]
            desc=[]
            for i in answers:
                # print(i)
                desc.append(i['n.name'])
                desc1.append(i['n.name'])
                drugs[i['n.name']]=i['id(n)']
            # print(drugs)
            # drugs =[i['id(n)'] for i in answers]
            temp = []
            
            for key, val in drugs.items():
                if val not in temp:
                    temp.append(val)
                    resultant_dictionary[key] = val
            print(resultant_dictionary)
            print(desc1)
            assert len(resultant_dictionary)==len(list(set(desc1)))
            # response = response + '已知患者可能的疾病为：{0}，可推荐使用的药品包括：{1}，对应药品id分别为{2}，请分情况推荐药品和药品id；'.format(final_wds_, '；'.join(list(set(desc))[:num_limit]),'；'.join(str(drugs.get(i)) for i in list(set(desc))[:num_limit]))
            response = response + '已知患者可能的疾病为：{0}，可推荐使用的药品包括：{1}，请分情况推荐药品；'.format(final_wds_, '；'.join(list(set(desc))[:num_limit]))

        print(response)
        return response

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
def generate_prompt(query, history, eos):
    global qu
    if not history:
        qu=False    
        return f"""现在是一位用户和智能医疗大模型FindGPT之间的对话。对于用户的医疗问诊，FindGPT给出准确的、详细的指导建议。对于用户的指令问题，FindGPT给出详细的、有礼貌的回答。<病人>：{query} <FindGPT>："""
    else:
        prompt = '现在是一位用户和智能医疗大模型FindGPT之间的对话。对于用户的医疗问诊，FindGPT会给出准确的、详细的指导建议。对于用户的指令问题，FindGPT给出详细的、有礼貌的回答。'
        # print(prompt)
        # print(type(query))
        qu=False
        for wd in query:
            if wd in drug_qwds:
                qu=True
        for i, (old_query, response) in enumerate(history):
            if qu:
                response=add_medical(response)
            prompt += "<病人>：{} <FindGPT>：{}".format(old_query, response) + eos
        prompt += "<病人>：{} <FindGPT>：".format(query)
        print("prompt:{}".format(prompt))
        return prompt

app = FastAPI()
@torch.inference_mode()
def chat_stream(model, tokenizer, query, history, max_new_tokens=512,
                temperature=0.2, repetition_penalty=1.2, context_len=1024, stream_interval=2):
        prompt = generate_prompt(query, history, tokenizer.eos_token)
        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        device = model.device
        stop_str = tokenizer.eos_token
        stop_token_ids = [tokenizer.eos_token_id]
        l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        for i in range(max_new_tokens):
            if i == 0:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=False)

                if stop_str:
                    pos = output.rfind(stop_str, l_prompt)
                    if pos != -1:
                        output = output[l_prompt:pos]
                        stopped = True
                    else:
                        output = output[l_prompt:]
                    yield output
                else:
                    raise NotImplementedError

            if stopped:
                break

        del past_key_values

@app.post("/chat")
async def create_item(request: Request):
    global model, tokenizer
    global desc1
    global resultant_dictionary
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('prompt')
    history = json_post_list.get('history')

    while True:
        pre = 0
        for outputs in chat_stream(model, tokenizer, query=query, history=history, max_new_tokens=1024,
                    temperature=0.5, repetition_penalty=1.2, context_len=1024):
                outputs = outputs.strip()
                # outputs = outputs.split("")
                now_1 = len(outputs)
                if now_1 - 1 > pre:
                    print(outputs[pre:now_1 - 1], end="", flush=True)
                    pre = now_1 - 1
        print(outputs[pre:], flush=True)
        print("outputs:{}".format(outputs))
        if outputs[0]=="：" or outputs[0]==":":
            outputs=outputs[1:]
            # print("outputs:{}".format(outputs))
        history = history + [(query, outputs)]
        global qu
        list_test=[]
        if qu:
            # print("test")
            # print(outputs)
            
            for drug in list(set(desc1)):
                # print(drug)
                if outputs.find(drug)!=-1:
                    # if drug in test.keys() and test.get(drug) is not None:
                        # outputs=outputs[:outputs.find(drug)]+drug+",链接为："+test[drug]+" "+outputs[outputs.find(drug)+len(drug):]
                    if drug in drugs.keys() and  drugs.get(drug) is not None:
                        # print(type(drugs.get(drug)))
                        list_test.append(drugs.get(drug))


        now = datetime.datetime.now()
        if outputs !=None:
            msg="success"
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "data":{
                "message": outputs,
                "history": history,
                "medicines": list_test
            },
            "code":1,
            "msg":msg
        }
        log = "[" + time + "] " + '", prompt:"' + query + '", response:"' + repr(outputs) + '"'
        print(log)
        # torch_gc()

        return answer

@app.post("/medicineInfo")
async def create_item(request: Request):
# 字段方面就分成
# id name type price info image
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    medicineIds = json_post_list.get('medicineIds')
    sql=[]
    answers=[]
    drug_info={}
    for i in medicineIds:
        sql1 = ["MATCH (n:Drug)-[r1:type_of]->(t:type), \
                (n:Drug)-[r2:price_of]->(p:price), \
                (n:Drug)-[r3:info_of]->(i:info), \
                (n:Drug)-[r4:url_of]->(u:urlist) \
                where id(n)=%d \
                return id(n),n.name,p.info,t.info,i.info,u.url" % (i)]
        sql=sql1
        for sql_ in sql:
            ress=g.run(sql_).data()
            if len(ress)!=0:
                answers+=ress
            else:
                ress=[{'id(n)':i,'n.name':'','p.info':'Not found','t.info':'','i.info':'','u.url':''}]
                answers+=ress
    for answ in answers:
        # print(answ)
        answ['id']=answ.pop('id(n)')
        answ['name']=answ.pop('n.name')
        answ['type']=answ.pop('p.info')
        answ['price']=answ.pop('t.info')
        answ['info']=answ.pop('i.info')
        answ['image']=answ.pop('u.url')
        drug_info[str(answ.get('id'))]=answ
    answer ='{ "data":{ "medicineInfo": '+str(drug_info)+'  },"code" : 1,"msg"  : "success"}'
    # data = json.loads(answer)
    # print(data)
    # data_=json.dumps(answer, sort_keys=True, indent=4, separators=(',',':'))
    data_=ast.literal_eval(answer)
    return data_

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    tokenizer = AutoTokenizer.from_pretrained("/home/username/FindGPT/model/FindGPT", padding_side="right", use_fast=True,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/username/FindGPT/model/FindGPT", low_cpu_mem_usage=True,offload_folder="offload",trust_remote_code=True)
    if torch.cuda.is_available():

        model.cuda()
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=10132, workers=1)