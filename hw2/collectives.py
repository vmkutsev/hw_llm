import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.ndim != 1 or tensor.numel() % world_size:
        raise ValueError("Expected a 1D tensor with length divisible by world_size")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    previous_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    # The last received and summed chunk must belong to this rank.
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        receive_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[receive_index])
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], next_rank),
            dist.P2POp(dist.irecv, received, previous_rank),
        ])
        for request in requests:
            request.wait()
        chunks[receive_index].add_(received)
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    previous_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_index], next_rank),
            dist.P2POp(dist.irecv, chunks[receive_index], previous_rank),
        ])
        for request in requests:
            request.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
