"""Real optional inference clients. No credential, download or inference fallback."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import urllib.request
from urllib.parse import urlparse
from .llm import Completion
from ..core.contracts import digest
from ..core.protocol import CapabilityUnsupported, ProtocolError


class TokenLedger:
    """Reserve before network/compute; ambiguous failures consume the reservation."""
    def __init__(self, limit):
        if type(limit) is not int or limit < 1: raise ValueError("token limit")
        self.limit, self.used = limit, 0
        self.lock = threading.Lock()

    def reserve(self, amount):
        if type(amount) is not int or amount < 1: raise ValueError("token reservation")
        with self.lock:
            if self.used + amount > self.limit: raise ProtocolError("MODEL_TOKEN_BUDGET_EXHAUSTED")
            self.used += amount
        return amount

    def settle(self, reserved, actual):
        if type(actual) is not int or not 0 <= actual <= reserved:
            raise ProtocolError("MODEL_USAGE_EXCEEDED_RESERVATION")
        with self.lock: self.used -= reserved - actual


class HTTPModelClient:
    """OpenAI-compatible Chat Completions with explicit input-token upper bound.

    input_bound must bound the provider's ENTIRE serialized prompt, including its
    framing. It is supplied explicitly rather than guessing tokens from characters.
    HTTPS required except loopback protocol tests. Redirects are disabled.
    """
    def __init__(self, model, base_url, key_env, ledger, input_bound, proxy=None):
        url = urlparse(base_url)
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("HTTPS endpoint required")
        if url.username or url.password or url.query or url.fragment: raise ValueError("endpoint credentials/query forbidden")
        if type(input_bound) is not int or input_bound < 1: raise ValueError("explicit provider prompt upper bound required")
        self.model, self.url, self.key_env = model, base_url.rstrip('/')+'/chat/completions', key_env
        self.ledger, self.input_bound, self.proxy = ledger, input_bound, proxy
        self.calls = []

    def __call__(self, prompt, max_tokens):
        key = os.environ.get(self.key_env)
        if not key: raise CapabilityUnsupported("CAPABILITY_UNSUPPORTED: missing model credential environment variable")
        body = {"model": self.model, "messages": [{"role":"user", "content":prompt}], "max_completion_tokens":max_tokens}
        reservation = self.ledger.reserve(self.input_bound + max_tokens)
        entry = {"request":body, "reservation":reservation, "status":"STARTED"}
        self.calls.append(entry)
        try:
            raw = json.dumps(body).encode()
            if self.proxy and self.proxy.startswith(('socks5://','socks5h://')):
                try: import httpx
                except ImportError: raise CapabilityUnsupported("CAPABILITY_UNSUPPORTED: install model-http extra for SOCKS")
                with httpx.Client(proxy=self.proxy.replace('socks5h://','socks5://'), timeout=120, follow_redirects=False) as client:
                    response=client.post(self.url, content=raw, headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
                    response.raise_for_status()
                    if len(response.content)>2000000: raise ProtocolError("MODEL_RESPONSE_SIZE")
                    data=response.json()
            else:
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *args, **kwargs): return None
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({"http":self.proxy,"https":self.proxy} if self.proxy else {}),NoRedirect())
                req=urllib.request.Request(self.url,raw,{"Authorization":"Bearer "+key,"Content-Type":"application/json"})
                with opener.open(req,timeout=120) as response:
                    payload=response.read(2000001)
                if len(payload)>2000000: raise ProtocolError("MODEL_RESPONSE_SIZE")
                data=json.loads(payload)
            entry['response']=data
            usage=data['usage']; inp,out=usage['prompt_tokens'],usage['completion_tokens']
            if type(inp) is not int or type(out) is not int or inp<0 or out<0 or inp>self.input_bound or out>max_tokens:
                raise ProtocolError("MODEL_USAGE_INVALID")
            self.ledger.settle(reservation,inp+out)
            choice=data['choices'][0]
            content=choice['message'].get('content')
            if not isinstance(content,str): raise ProtocolError("MODEL_REFUSAL_OR_NON_TEXT")
            entry['status']='COMPLETE'
            return Completion(content,self.model,inp,out,choice['finish_reason']!='stop')
        except Exception as exc:
            # Exception text may contain URLs/credentials supplied by a transport.
            entry['status']='FAILED'; entry['error_type']=type(exc).__name__
            raise


def validate_checkpoint(path, role, base_model):
    from hashlib import sha256
    path=Path(path).resolve()
    data=json.loads(path.read_text())
    if data['role']!=role or data['base_model']!=base_model: raise ProtocolError("checkpoint identity mismatch")
    for name, expected in data['files'].items():
        file=(path.parent/name).resolve()
        if not file.is_relative_to(path.parent) or not file.is_file() or sha256(file.read_bytes()).hexdigest()!=expected:
            raise ProtocolError("checkpoint checksum/path mismatch")
    if not data['files'] or 'adapter_config.json' not in data['files']: raise ProtocolError("missing adapter payload")
    return data


class LocalHFClient:
    """Role-owned HF model, local assets only, actual PEFT checkpoint loading."""
    def __init__(self, model, role, ledger, checkpoint=None, device='cpu'):
        self.model,self.role,self.ledger,self.checkpoint,self.device=model,role,ledger,checkpoint,device
        self.calls=[]; self.loaded=None

    def prepare(self, state):
        if state.role!=self.role or state.base_model!=self.model: raise ProtocolError("local model identity mismatch")
        if state.delta_checkpoint!=self.checkpoint:
            self.checkpoint=state.delta_checkpoint; self.loaded=None

    def load(self):
        if self.loaded is not None: return self.loaded
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError: raise CapabilityUnsupported("CAPABILITY_UNSUPPORTED: install training extra and provision local model assets")
        tokenizer=AutoTokenizer.from_pretrained(self.model,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(self.model,local_files_only=True,trust_remote_code=False).to(self.device)
        if self.checkpoint:
            validate_checkpoint(self.checkpoint,self.role,self.model)
            try: from peft import PeftModel
            except ImportError: raise CapabilityUnsupported("CAPABILITY_UNSUPPORTED: peft unavailable")
            model=PeftModel.from_pretrained(model,str(Path(self.checkpoint).parent),is_trainable=False,local_files_only=True)
        model.eval(); self.loaded=(tokenizer,model)
        return self.loaded

    def __call__(self,prompt,max_tokens):
        import math
        tokenizer,model=self.load()
        import torch
        encoded=tokenizer(prompt,return_tensors='pt').to(self.device)
        n=encoded['input_ids'].shape[1]
        reserved=self.ledger.reserve(n+max_tokens)
        with torch.no_grad():
            result=model.generate(**encoded,max_new_tokens=max_tokens,do_sample=True,top_k=0,top_p=1.0,temperature=1.0,return_dict_in_generate=True,output_scores=True,pad_token_id=tokenizer.eos_token_id)
        ids=result.sequences[0,n:]; count=len(ids)
        self.ledger.settle(reserved,n+count)
        logprob=sum(float(torch.log_softmax(s[0],dim=-1)[ids[i]]) for i,s in enumerate(result.scores))
        text=tokenizer.decode(ids,skip_special_tokens=True)
        completion=Completion(text,self.model,n,count,count==max_tokens,logprob,digest((self.model,self.checkpoint)),tuple(encoded["input_ids"][0].tolist()),tuple(ids.tolist()))
        self.calls.append({'request':prompt,'response':completion.__dict__,'role':self.role})
        return completion
