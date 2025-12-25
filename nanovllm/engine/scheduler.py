from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()  # running中的seqs

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """## Add a sequence to the waiting queue."""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        ## return
            list[Sequence]
            bool: is_prefill
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # waiting中有新的seqs且未达到支持的最大序列数量，则按先进先出策略
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 当总的tokens数+新的seq的tokens数大于max_num_batched_tokens,或block已满, 则不新增seq到running队列中
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 若有新增seq加入running队列，则return，当前为prefill
        if scheduled_seqs:
            return scheduled_seqs, True
        # 若未return, 则说明无新seq，继续做decode
        while self.running and num_seqs < self.max_num_seqs:
            # 不断将running中的seq拿出来处理,分配block
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # block已满，若running不为空则剔除最后一个seq;为空则剔除当前seq并跳出
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))  # 将调度好的倒序后再反过来被拓展到running队列前面
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        将seq重新放入waiting队列
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
