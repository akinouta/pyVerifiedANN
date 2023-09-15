import heapq

candidate = []
heapq.heappush(candidate, (12312.312, 5))
i,j=heapq.heappop(candidate)
print(i,j)
