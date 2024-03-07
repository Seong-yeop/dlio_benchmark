import ctypes
import logging

class ConsistentHashingWrapper(ctypes.Structure):
    _fields_ = [
        ("instance", ctypes.c_void_p)  # Pointer to ConsistentHashing instance
    ]

class SizedBuf(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_char_p),  
        ("size", ctypes.c_size_t)
    ]

class Doc(ctypes.Structure):
    _fields_ = [
        ("id", SizedBuf),
        ("data", SizedBuf)
    ]

class DbHandle(ctypes.Structure):
    _fields_ = [
        ("dbs", ctypes.POINTER(ctypes.c_void_p)),
        ("ch", ctypes.c_void_p),
        ("node_num", ctypes.c_int),
    ]

# Load the shared library
rocksdb_lib = ctypes.CDLL('/home/csl/testbed/rocksdb/librocksdb.so')  
#multi_lib = ctypes.CDLL('/home/csl/testbed/new-multi-node-Dotori/bench/rocksdb_test/libmulti_rocksdb.so')  
multi_lib = ctypes.CDLL('/home/csl/testbed/new-multi-node-Dotori/bench/dotori_test/libmulti_dotori.so')  

libc = ctypes.CDLL('libc.so.6')  

create_consistent_hasing_func = multi_lib.create_consistent_hashing
create_consistent_hasing_func.argtypes = [ctypes.c_int]
create_consistent_hasing_func.restype = ctypes.POINTER(ConsistentHashingWrapper)



class DbWrapper:
    __instance = None

    def __init__(self):
        self.handle = None
        self.dataset_key = []

        if DbWrapper.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DbWrapper.__instance = self
            self.db_init(4)
    
    def get_instance():
        if DbWrapper.__instance == None:
            DbWrapper()
        return DbWrapper.__instance

    def db_init(self, num_nodes, db_name="rocksdb", file_path="test_simple_dir/rocks/node"):
        dbname = db_name
        filename = file_path
        c_filename = ctypes.c_char_p(filename.encode('utf-8'))
        
        self.handle = DbHandle()
        self.handle.node_num = num_nodes
        self.handle.dbs = (ctypes.c_void_p * self.handle.node_num)()

        consistent_hash = create_consistent_hasing_func(32)

        for i in range(self.handle.node_num):
            #print("add node: ", i)
            multi_lib.add_node(consistent_hash, i)

        self.handle.ch = consistent_hash.contents.instance
        multi_lib.couchstore_open_multiple_dbs(c_filename, 1, ctypes.byref(self.handle))
        #print("open multiple dbs done")
    
    def db_close(self):
        multi_lib.couchstore_close_multiple_dbs(ctypes.byref(self.handle))

    def db_checkpoint(self, uri, data: str):
        bs = 8192
        size = len(data)//bs + 1
        docs = (Doc * size)()
        for i in range(size-1):
            #key = f"{uri}-{i}"
            key = f"c{i}"

            docs[i].id.buf = ctypes.c_char_p(key.encode('utf-8'))
            docs[i].id.size = len(key)
            
            value = data[i*bs:(i+1)*bs]
            docs[i].data.buf = ctypes.c_char_p(value.encode('utf-8'))
            docs[i].data.size = len(value)

            multi_lib.couchstore_save_document_to_nodes(ctypes.byref(self.handle), ctypes.byref(docs[i]), None, 0)

    def db_write(self, uri, data):
        docs = (Doc)()

        key = f"{uri}"
        docs.id.buf = ctypes.c_char_p(key.encode('utf-8'))
        docs.id.size = len(key)
            

        docs.data.buf = data
        docs.data.size = len(data)
        # print("[write] key: ", docs.id.buf, "value len: ", docs.data.size)
        
        multi_lib.couchstore_save_document_to_nodes(ctypes.byref(self.handle), ctypes.byref(docs), None, 0)
        self.dataset_key.append(key)
        multi_lib.couchstore_commit_nodes(ctypes.byref(self.handle))

    def db_read(self, uri, dataset):
        rq_doc = ctypes.POINTER(Doc)()
        key = f"{uri}"
        rq_id = SizedBuf()
        rq_id.buf = ctypes.c_char_p(key.encode('utf-8'))
        rq_id.size = len(key)

        multi_lib.couchstore_open_document_from_nodes(ctypes.byref(self.handle), rq_id.buf, rq_id.size, ctypes.byref(rq_doc), 0)

        dataset.append(rq_doc.contents.data.buf)
        # dataset.append(ctypes.string_at(rq_doc.contents.data.buf,rq_doc.contents.data.size).decode('utf-8'))
        # print("[read] key: ", rq_id.buf, "value len: ", rq_doc.contents.data.size)
        rq_doc.contents.id.buf = None
        multi_lib.couchstore_free_document(rq_doc)

        return rq_doc.contents.data.size



if __name__ == "__main__":
    dbname = "RocksDB"
    filename = "test_simple_dir/rocks/node"
    c_filename = ctypes.c_char_p(filename.encode('utf-8'))


    print("DB: ", dbname)
    print("Filename: ", filename)

    handle = DbHandle()
    handle.node_num = 2
    handle.dbs = (ctypes.c_void_p * handle.node_num)()

    consistent_hash = create_consistent_hasing_func(32)

    for i in range(handle.node_num):
        print("add node: ", i)
        multi_lib.add_node(consistent_hash, i)

    handle.ch = consistent_hash.contents.instance

    multi_lib.couchstore_open_multiple_dbs(c_filename, 1, ctypes.byref(handle))
    print("open multiple dbs done")

    batch_size = 32
    docs = (Doc * batch_size)()

    for i in range(batch_size):
        key = f"key{i}"
        docs[i].id.buf = ctypes.c_char_p(key.encode('utf-8'))
        docs[i].id.size = len(key)
        value = f"value{i}"
        docs[i].data.buf = ctypes.c_char_p(value.encode('utf-8'))
        docs[i].data.size = len(value) 
        print("key: ", docs[i].id.buf, "value: ", docs[i].data.buf)


    for i in range(batch_size):
        multi_lib.couchstore_save_document_to_nodes(ctypes.byref(handle), ctypes.byref(docs[i]), None, 0)
        print("save multiple documents done")

    multi_lib.couchstore_commit_nodes(ctypes.byref(handle))
    print("commit multiple dbs done")

    rq_doc = ctypes.POINTER(Doc)()

    for i in range(batch_size):
        key = f"key{i}"
        rq_id = SizedBuf()
        rq_id.buf = ctypes.c_char_p(key.encode('utf-8'))
        rq_id.size = len(key)    
        
        multi_lib.couchstore_open_document_from_nodes(ctypes.byref(handle), rq_id.buf, rq_id.size, ctypes.byref(rq_doc), 0)
        print(f"{rq_id.buf}: {rq_doc.contents.data.buf}")

    multi_lib.couchstore_close_multiple_dbs(ctypes.byref(handle))
    print("close multiple dbs done")
