class MemoryManager():
    def __init__(self):
        self.max_memory = 0
        self.free_list = []

        self.current_allocation_id = 1

    def allocate_memory(self, buffer_size):
        print("allocatiing", buffer_size)
        print(self.free_list)
        for block in self.free_list:
            if buffer_size <= block["size"]:
                if block["size"] != buffer_size:
                    new_block = dict(id = self.current_allocation_id,
                                     start = copy(block["start"]),
                                     size = buffer_size)
                    block["size"] -= buffer_size
                    block["start"] += buffer_size
                    print("block:", block)
                    print("new_block: ", new_block)
                    self.current_allocation_id += 1
                    return new_block
                else:
                    return block

        new_block = dict(id = self.current_allocation_id,
                         start = self.max_memory,
                         size = buffer_size)
        self.max_memory += buffer_size
        self.current_allocation_id += 1

        print("freshly allocated block", new_block)
        
        return new_block
        
    def free_memory(self, buffer):
        print("deallocating", buffer)
        def find_buffer(start, size):
            end = start + size
            for buffer in self.free_list:
                buffer_start = buffer["start"]
                buffer_end = buffer["start"] + buffer["size"]

                max_start = max(start, buffer_start)
                min_end = min(end, buffer_end)

                if max_start <= min_end:
                    return buffer

            return None

        overlapping_buffer = find_buffer(buffer ["start"], buffer["size"])
        while(overlapping_buffer):
            self.free_list.remove(overlapping_buffer)

            start = min(buffer["start"], overlapping_buffer["start"])
            end = max(buffer["start"]+buffer["size"], overlapping_buffer["start"]+overlapping_buffer["size"])

            buffer["start"] = start
            buffer["size"] = end - start

            overlapping_buffer = find_buffer(buffer["start"], buffer["size"])
            
        self.free_list.append(buffer)
        self.free_list.sort(key = lambda x : x["start"])

        print("Free list", self.free_list)
