#include <tensorflow/c/c_api.h>
#include <random>
#include <iostream>

static void NoOpDeallocator(void *data, size_t len, void *arg) {}

const char *tags = "serve";

void print_all_operations(const char *model_path) {
    TF_SessionOptions *session_options = TF_NewSessionOptions();
    TF_Status *status = TF_NewStatus();
    TF_Buffer *graph_def = TF_NewBuffer();
    TF_Graph *graph_ = TF_NewGraph();

    TF_Session *session_ = TF_LoadSessionFromSavedModel(session_options, nullptr, model_path,
                                                        &tags, 1, graph_, graph_def, status);

    size_t pos = 0;
    TF_Operation *oper;
    while ((oper = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
        std::cout << TF_OperationName(oper) << " " << TF_OperationOpType(oper) << std::endl;
    }


    //释放空间
    TF_DeleteGraph(graph_);
    TF_CloseSession(session_, status);
    TF_DeleteSession(session_, status);
    TF_DeleteStatus(status);
    TF_DeleteBuffer(graph_def);
    TF_DeleteSessionOptions(session_options);
}


int main(int argc, char **argv) {
    print_all_operations(argv[1]);
    return 0;
}