#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace std;

Module *makeLLVMModule() {
    // First, Create the module, set datalayout and target triple
    Module *mod = new Module("sum.ll", *LLVMGetGlobalContext());
    mod->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
    mod->setTargetTriple("x86_64-unknown-linux-gnu");

    // Configure the function signature
    SmallVector<Type*, 2> FuncTyArgs;
    FuncTyArgs.push_back(IntegerType::get(mod->getContext(), 32));
    FuncTyArgs.push_back(IntegerType::get(mod->getContext(), 32));
    FunctionType *FuncTy = FunctionType::get(IntegerType::get(mod->getContext(), 32), FuncTyArgs, false);

    // Create the function 
    Function *funcSum = Function::Create(FuncTy, GlobalValue::ExternalLinkage, "@_Z3sumii", mod);
    funcSum->setCallingConv(CallingConv::C);

    // Create Value pointer to save args
    Function::arg_iterator args = funcSum->arg_begin();
    Value *int32_0 = args++;
    int32_0->setName("0");
    Value *int32_1 = args++;
    int32_1->setName("1");

    // Create the first basic block with entry label, then add instructions to the block
    BasicBlock *labelEntry = BasicBlock::Create(mod->getContext(), "", funcSum, 0);

    // Instructions
    AllocaInst *ptrA = new AllocaInst(IntegerType::get(mod->getContext(), 32), "3", labelEntry);
    ptrA->setAlignment(4);
    AllocaInst *ptrB = new AllocaInst(IntegerType::get(mod->getContext(), 32), "4", labelEntry);
    ptrB->setAlignment(4);

    StoreInst *st0 = new StoreInst(int32_0, ptrA, false, labelEntry);
    st0->setAlignment(4);
    StoreInst *st1 = new StoreInst(int32_1, ptrB, false, labelEntry);
    st1->setAlignment(4);

    LoadInst *ld0 = new LoadInst(ptrA, "", false, labelEntry);
    ld0->setAlignment(4);
    LoadInst *ld1 = new LoadInst(ptrB, "", false, labelEntry);
    ld1->setAlignment(4);

    BinaryOperator *addRes = BinaryOperator::Create(Instruction::Add, ld0, ld1, "add", labelEntry);
    ReturnInst::Create(mod->getContext(), addRes, labelEntry);

    return mod;
}

int main() {
    Module *Mod = makeLLVMModule();
    verifyModule(*Mod, PrintMessage);
    string ErrorInfo;
    OwningAtomPtr<tool_output_file> Out(new tool_output_file("./sum.bc", ErrorInfo, sys::fs::F_None));
    
    if (!ErrorInfo.empty()) {
        errs() << ErrorInfo << '\n';
        return -1;
    }

    WriteBitcodeToFile(Mod, Out->os());
    Out->keep();

    return 0;
}
