import os;
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import time
from chunked_engine import Engine
from chunked_scheduler import Scheduler, InputRequest

engine = Engine()
ids = engine.tokenizer(' '.join(['hello'] * 600), return_tensors='pt').input_ids[0]
prompt = engine.tokenizer.decode(ids[:512], skip_special_tokens=True)

scheduler = Scheduler(engine, token_batch_size=512)
for _ in range(100):
    scheduler.add_req(InputRequest(prompt, output_len=512))

start = time.time()
while not scheduler.finished():
    scheduler.run()
elapsed = time.time() - start
print(f'ETE Time: {elapsed:.4f} s')
print(f'Output throughput: {100*512/elapsed:.2f} tokens/s')
