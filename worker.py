"""Colab worker: install repo, set HUB_URL and WORKER_TOKEN, run.
Model generates prompt + analysis metadata; original text is completion.
"""
import argparse, os, time, uuid, json, re, threading, traceback, hashlib
from pathlib import Path
import requests

# Local status file: no Hub URL or token is written to disk.
STATUS_PATH=Path(os.environ.get('STORYHUB_STATUS_FILE','/content/storyhub-worker-status.json'))
STATUS_LOCK=threading.Lock()
STATUS={"stage":"starting","worker_id":"","gpu":"","model":"","story_id":"","story_title":"","chapter":0,"total_chapters":0,"samples":0,"error":"","updated_at":0}
def report_status(**values):
    with STATUS_LOCK:
        STATUS.update(values,updated_at=time.time())
        try:
            STATUS_PATH.parent.mkdir(parents=True,exist_ok=True)
            temp=STATUS_PATH.with_suffix('.tmp')
            temp.write_text(json.dumps(STATUS,ensure_ascii=False),encoding='utf-8')
            os.replace(temp,STATUS_PATH)
        except OSError:pass

RULES='''Bạn là bộ phân tích truyện để xây dựng dữ liệu SFT tiếng Việt.
Chỉ trả về một đối tượng JSON có các khóa genre, style, instruction, summary.
Dựa hoàn toàn vào nội dung ngữ cảnh và chương được cung cấp.
Không tiết lộ sự kiện phía sau đoạn hoàn thiện. Không nêu tên tác giả/văn phong đặc trưng của tác giả trong instruction.
Instruction mô tả đề bài sáng tác ngắn (thể loại, cảnh, nhân vật, cảm xúc), không chép nguyên completion.
Không sửa nội dung truyện, không bịa chi tiết chưa biết. Nếu thông tin không rõ, dùng nhãn 'không xác định'.
Không trích các đoạn dài từ completion vào instruction.
Không dùng markdown fences.'''

def safe_json(text):
    try: return json.loads(text)
    except Exception:
        try:
            m=re.search(r'\{.*\}',text,re.S)
            return json.loads(m.group()) if m else {}
        except Exception:return {}

class Model:
    def __init__(self,model_id,mode,rules=RULES):
        self.mode=mode;self.id=model_id;self.rules=rules;self.pipe=None
        if mode=='transformers':
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
            import torch
            kw={'device_map':'auto'}
            if torch.cuda.is_available():
                kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
            self.tokenizer=AutoTokenizer.from_pretrained(model_id,trust_remote_code=False)
            m=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,**kw)
            self.pipe=pipeline('text-generation',model=m,tokenizer=self.tokenizer)
    def analyze(self,context,text,title):
        if self.mode=='mock':
            return {'genre':'chưa xác định','style':'tự sự','summary':'Đoạn văn tiếp nối các sự kiện đã cho','instruction':f'Viết tiếp một cảnh thuộc truyện {title}, giữ nguyên nhân vật và không khí đã nêu trong phần ngữ cảnh.'}
        prompt=f'''Tên truyện: {title}\nNgữ cảnh trước đoạn cần viết:\n{context[-2200:]}\nĐoạn cần phân tích (chỉ đọc, không viết lại):\n{text[:3500]}\n\nTrả JSON với genre, style, summary, instruction.'''
        messages=[{'role':'system','content':self.rules},{'role':'user','content':prompt}]
        if self.mode=='ollama':
            url=os.getenv('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')+'/api/chat'
            r=requests.post(url,json={'model':self.id,'messages':messages,'format':'json','stream':False,'options':{'temperature':0.2,'num_predict':400}},timeout=180)
            r.raise_for_status();output=r.json()['message']['content']
        else:
            inputs=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            res=self.pipe(inputs,max_new_tokens=400,do_sample=False,return_full_text=False)
            output=res[0]['generated_text']
        return safe_json(output)

class Client:
    def __init__(self,url,token):
        self.base=url.rstrip('/');self.s=requests.Session();self.s.headers.update({'Authorization':'Bearer '+token})
    def get(self,path,**params):
        r=self.s.get(self.base+path,params=params,timeout=120);r.raise_for_status();return r.json()
    def post(self,path,body):
        r=self.s.post(self.base+path,json=body,timeout=120);r.raise_for_status();return r.json()

def make_samples(model,story,chapters,chapter_index,limit_per_story):
    ch=chapters[chapter_index]
    # Use bounded target segments so they fit a 1024-2048 token SFT context.
    content=ch['content'].strip()
    segments=[]
    remainder=content
    while remainder:
        if len(remainder)<=2000:
            chunk=remainder
            remainder=''
        else:
            cut=remainder.rfind('\n\n',650,2000)
            if cut<650:
                cut=max(remainder.rfind('. ',800,2000),remainder.rfind('… ',800,2000),remainder.rfind('! ',800,2000))
            if cut<650:cut=1800
            chunk=remainder[:cut]
            remainder=remainder[cut:].lstrip()
        chunk=chunk.strip()
        if chunk:segments.append(chunk)
    prev=chapters[chapter_index-1]['content'][-1000:] if chapter_index else story.get('description','')
    result=[]
    for seg_index,seg in enumerate(segments):
        if len(seg)<80:continue
        context=prev if seg_index==0 else segments[seg_index-1][-1000:]
        info=model.analyze(context,seg,story['title'])
        instruction=str(info.get('instruction') or '').strip()
        if len(instruction)<30: instruction=f'Viết tiếp cảnh truyện {story["title"]}, giữ nhất quán nhân vật và bối cảnh.'
        instruction=instruction[:1400]
        user=f'{instruction}\n\nNgữ cảnh trước đó:\n{context[-1100:]}'
        uid=hashlib.sha256(f'{story["id"]}|{ch["chapter_id"]}|{seg_index}|{ch["content_hash"]}'.encode()).hexdigest()[:28]
        result.append({'sample_id':'sft_'+uid,'chapter_id':ch['chapter_id'],'task':'story_continuation',
          'prompt':[{'role':'system','content':'Bạn là nhà văn sáng tác truyện tiếng Việt. Viết đoạn truyện mới theo yêu cầu, tự nhiên và nhất quán.'},{'role':'user','content':user}],
          'completion':[{'role':'assistant','content':seg}], 'source_hash':ch['content_hash']})
        if len(result)>=max(1,limit_per_story):break
    return result

def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            p=torch.cuda.get_device_properties(0)
            return p.name,round(p.total_memory/1024**3,1)
    except Exception:pass
    return 'CPU',0

def run(args):
    client=Client(args.hub_url,args.token)
    name,vram=gpu_info()
    worker_id=os.getenv('WORKER_ID') or 'colab-'+uuid.uuid4().hex[:12]
    report_status(stage='registering',worker_id=worker_id,gpu=name,model=args.model)
    client.post('/api/worker/register',{'worker_id':worker_id,'name':args.name or worker_id,
                'gpu':name,'vram_gb':vram,'model':args.model})
    print('Kết nối Hub thành công:',worker_id,'GPU:',name,'đang tải model...',flush=True)
    report_status(stage='loading_model')
    config=client.get('/api/worker/config')
    loading=threading.Event()
    def loading_pulse():
        while not loading.is_set():
            try:client.post('/api/worker/state',{'worker_id':worker_id,'status':'loading_model',
                 'detail':{'stage':'loading_model','model':args.model}})
            except Exception:pass
            loading.wait(20)
    pulse=threading.Thread(target=loading_pulse,daemon=True);pulse.start()
    try:
        model=Model(args.model,args.engine,rules=config['rules']+'\n'+config['skill'])
    except Exception as ex:
        report_status(stage='error',error='Không tải được model: '+str(ex)[:300])
        try:client.post('/api/worker/state',{'worker_id':worker_id,'status':'error','detail':{'stage':'loading_model','error':str(ex)[:200]}})
        except Exception:pass
        raise
    finally:
        loading.set();pulse.join(timeout=3)
    report_status(stage='idle',error='')
    client.post('/api/worker/state',{'worker_id':worker_id,'status':'idle',
                'detail':{'stage':'ready','model':args.model}})
    print('Model sẵn sàng. Rules hash:',config['hash'][:12],flush=True)
    while True:
        try:
            response=client.post('/api/worker/claim',{'worker_id':worker_id})
            job=response.get('job')
            if not job:
                report_status(stage='idle',story_id='',story_title='')
                time.sleep(args.poll);continue
            sid=job['story_id'];attempt=job['attempt_id'];token=job['lease_token']
            report_status(stage='claimed',story_id=sid,chapter=0,total_chapters=0,samples=0)
            common={'worker_id':worker_id,'story_id':sid,'attempt_id':attempt,'lease_token':token}
            stop=threading.Event();progress={'n':0,'total':0,'stage':'starting','samples':0}
            def beat():
                while not stop.is_set():
                    try:
                        client.post('/api/worker/heartbeat',{
                            **common,'progress':progress['n'],
                            'detail':{'chapter':progress['n'],'total':progress['total'],
                                      'stage':progress['stage'],'samples':progress['samples'],'model':args.model}})
                    except Exception as ex:print('heartbeat:',str(ex)[:100],flush=True)
                    stop.wait(20)
            thread=threading.Thread(target=beat,daemon=True);thread.start()
            try:
                payload=client.get('/api/worker/story/'+sid,attempt_id=attempt,lease_token=token)
                story=payload['story'];n=payload['total_chapters']
                report_status(stage='processing',story_title=story['title'],total_chapters=n)
                completed=set(payload.get('completed_chapters',[]))
                progress['total']=n
                remaining=max(0,job['max_samples']-payload.get('sample_count',0))
                print('Xử lý:',sid,story['title'],'chương:',n,'mẫu còn:',remaining,flush=True)
                previous='';summaries=[];added=0
                for offset in range(0,n,args.batch_chapters):
                    progress['stage']='downloading'
                    report_status(stage='downloading')
                    page=client.get('/api/worker/story/'+sid+'/chapters',attempt_id=attempt,
                                    lease_token=token,offset=offset,limit=args.batch_chapters)
                    chapters=page['chapters']
                    for ch in chapters:
                        progress['n']=ch['chapter_index']
                        content=ch['content']
                        if ch['chapter_id'] in completed or ch.get('checkpoint')=='done':
                            previous=content[-1100:];continue
                        progress['stage']='analyzing'
                        report_status(stage='analyzing',chapter=ch['chapter_index'])
                        # Keep only a current chapter in memory; no full-book payload.
                        local_story={**story,'description':previous or story.get('description','')}
                        batch=make_samples(model,local_story,[ch],0,min(remaining,args.samples_per_chapter)) if remaining else []
                        if batch:
                            progress['stage']='uploading'
                            report_status(stage='uploading')
                            resp=client.post('/api/worker/samples',{**common,'samples':batch})
                            added+=resp['accepted'];remaining=max(0,remaining-resp['accepted'])
                            progress['samples']=added
                            report_status(samples=added)
                            summaries.append({'chapter_id':ch['chapter_id'],'index':ch['chapter_index'],
                                              'preview':content[:250]})
                        # Persist completion regardless of whether the chapter created samples.
                        client.post('/api/worker/checkpoint',{**common,'chapter_id':ch['chapter_id'],
                           'source_hash':ch['content_hash'],'rules_hash':config['hash'],
                           'state':'done','samples_generated':len(batch)})
                        previous=content[-1100:]
                        report_status(stage='checkpointed',chapter=ch['chapter_index'],samples=added)
                    print('Truyện',sid,'chương',min(offset+len(chapters),n),'/',n,'mẫu mới',added,flush=True)
                if summaries:
                    client.post('/api/worker/profile',{**common,'model_id':args.model,
                       'rules_hash':config['hash'],'profile':{'story_id':sid,'title':story['title'],
                       'chapters_observed':summaries[-30:],'note':'Bản tóm tắt cấp chương, không phải phân tích toàn bộ cốt truyện'}})
                progress['stage']='complete'
                report_status(stage='completed',chapter=n,samples=added)
                result=client.post('/api/worker/complete',common)
                print('HOÀN TẤT',sid,result,flush=True)
            except Exception as ex:
                report_status(stage='error',error=str(ex)[:400])
                print('Lỗi job',sid,repr(ex),flush=True)
                try:client.post('/api/worker/fail',{**common,'message':str(ex)[:480]})
                except Exception:pass
            finally:stop.set();thread.join(timeout=3)
        except KeyboardInterrupt:
            print('Dừng worker');break
        except Exception as ex:
            report_status(stage='disconnected',error=str(ex)[:400]);print('Mất kết nối:',repr(ex),flush=True);time.sleep(args.poll)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--hub-url',default=os.getenv('HUB_URL',''))
    p.add_argument('--token',default=os.getenv('WORKER_TOKEN',''))
    p.add_argument('--model',default=os.getenv('MODEL_ID','Qwen/Qwen2.5-3B-Instruct'))
    p.add_argument('--engine',choices=['transformers','ollama','mock'],default=os.getenv('ENGINE','transformers'))
    p.add_argument('--name',default='')
    p.add_argument('--poll',type=int,default=10)
    p.add_argument('--batch-chapters',type=int,default=10)
    p.add_argument('--samples-per-chapter',type=int,default=3)
    a=p.parse_args()
    if not a.hub_url or not a.token:p.error('HUB_URL and WORKER_TOKEN required')
    run(a)
