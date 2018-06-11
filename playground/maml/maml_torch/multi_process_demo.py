
def fn():
    pass
    
if __name__ == '__main__':
    model = Net()
    model.share_memory()
    processes = []
    for rank in range(10):
        p = mp.Process(target=fn, args=(rank, args, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

        
