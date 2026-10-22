"""Optional actual role-isolated LoRA SFT / clipped importance policy gradient.

RL is an offline single update per behavior batch, not PPO or a simulated run.
Only train rollouts with independently measured rewards are admitted.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from pathlib import Path
from ..core.contracts import ModelState, digest
from ..core.protocol import CapabilityUnsupported, ProtocolError
from .models import validate_checkpoint


@dataclass(frozen=True)
class TrainingRecord:
    prompt: str
    completion: str
    evidence_hashes: tuple
    reward: float
    split: str = 'train'
    behavior_logprob: float = None
    policy_id: str = None
    prompt_ids: tuple = ()
    completion_ids: tuple = ()


def validate_records(records, batch, algorithm):
    if not records: raise ProtocolError('no independently evaluated train rollouts')
    for r in records:
        if r.split!='train' or not r.evidence_hashes or not set(r.evidence_hashes)<=set(batch.evidence_hashes):
            raise PermissionError('training rollout outside verified train evidence')
        if not r.prompt or not r.completion or not math.isfinite(r.reward) or not 0<=r.reward<=1:
            raise ProtocolError('invalid training record')
        if algorithm=='rl' and (r.behavior_logprob is None or not math.isfinite(r.behavior_logprob) or not r.policy_id or not r.prompt_ids or not r.completion_ids):
            raise ProtocolError('RL requires actual behavior likelihood and policy identity')


class LoRABackend:
    def __init__(self, records, algorithm='sft', steps=1, rank=8, learning_rate=1e-4, device='cpu', max_sequence=2048):
        if algorithm not in ('sft','rl') or not 1<=steps<=10000 or not 1<=rank<=256 or not 0<learning_rate<=.1:
            raise ValueError('training configuration bounds')
        self.records=tuple(records); self.algorithm=algorithm; self.steps=steps; self.rank=rank
        self.learning_rate=learning_rate; self.device=device; self.max_sequence=max_sequence; self.receipts=[]

    def fit(self, state, batch, output_directory):
        started=time.perf_counter()
        validate_records(self.records,batch,self.algorithm)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, PeftModel
        except ImportError: raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: torch/transformers/peft and local model assets required')
        # A fresh base per role/fit prevents PEFT's in-place mutation leaking weights.
        tokenizer=AutoTokenizer.from_pretrained(state.base_model,local_files_only=True,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(state.base_model,local_files_only=True,trust_remote_code=False).to(self.device)
        if state.delta_checkpoint:
            validate_checkpoint(state.delta_checkpoint,state.role,state.base_model)
            model=PeftModel.from_pretrained(base,str(Path(state.delta_checkpoint).parent),is_trainable=True,local_files_only=True)
        else:
            model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=self.rank,lora_alpha=2*self.rank,lora_dropout=0.0,target_modules='all-linear'))
        parameters=[p for p in model.parameters() if p.requires_grad]
        if not parameters: raise ProtocolError('no trainable LoRA parameters')
        before=[p.detach().clone() for p in parameters]
        optimizer=torch.optim.AdamW(parameters,lr=self.learning_rate)
        losses=[]; tokens=0
        model.train() if self.algorithm=="sft" else model.eval()
        for i in range(self.steps):
            r=self.records[i%len(self.records)]
            prefix=tokenizer.encode(r.prompt,add_special_tokens=True)
            suffix=tokenizer.encode(r.completion,add_special_tokens=False)
            if r.completion_ids:
                if tuple(prefix)!=r.prompt_ids or tokenizer.decode(r.completion_ids,skip_special_tokens=True)!=r.completion:
                    raise ProtocolError('recorded tokenization mismatch')
                suffix=list(r.completion_ids)
            if not suffix or len(prefix)+len(suffix)>self.max_sequence: raise ProtocolError('training sequence budget exceeded; no silent truncation')
            ids=torch.tensor([prefix+suffix],device=self.device)
            labels=ids.clone(); labels[:,:len(prefix)]=-100
            optimizer.zero_grad()
            if self.algorithm=='sft':
                # Reward-weighted SFT; zero-reward invalid candidates never train.
                if r.reward<=0: raise ProtocolError('SFT requires independently useful rollout')
                loss=model(input_ids=ids,labels=labels).loss*r.reward
            else:
                expected=digest((state.base_model,state.delta_checkpoint))
                if r.policy_id!=expected: raise ProtocolError('RL behavior policy mismatch')
                logits=model(input_ids=ids).logits[:,:-1,:]
                lp=torch.log_softmax(logits,dim=-1).gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
                current=lp[:,max(0,len(prefix)-1):].sum()
                ratio=torch.exp(torch.clamp(current-r.behavior_logprob,-20,20))
                loss=-torch.clamp(ratio,max=2.0)*r.reward
            if not torch.isfinite(loss): raise ProtocolError('nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach())); tokens+=len(prefix)+len(suffix)
        if not any(not torch.equal(a,b) for a,b in zip(before,parameters)): raise ProtocolError('training made no parameter update')
        output=Path(output_directory)
        model.save_pretrained(output,safe_serialization=True)
        files={str(p.relative_to(output)):sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
        receipt={'role':state.role,'base_model':state.base_model,'training_data_hash':digest(batch),
                 'records_hash':digest(self.records),'algorithm':self.algorithm,'steps':self.steps,'tokens':tokens,
                 'losses':losses,'device':self.device,'elapsed_seconds':time.perf_counter()-started,'files':files,'parent_checkpoint':state.delta_checkpoint}
        manifest=output/'manifest.json'; manifest.write_text(json.dumps(receipt,indent=2))
        self.receipts.append(receipt)
        return ModelState(state.base_model,state.role,str(manifest.resolve()),digest(batch))
