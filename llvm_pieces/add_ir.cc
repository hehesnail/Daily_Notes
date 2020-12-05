#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm-c/Core.h"
#include "llvm/Support/raw_os_ostream.h"
#include <iostream>

using namespace llvm;
using namespace std;

Module *makeLLVMModule() {
    // First, Create the module, set datalayout and target triple
    Module *mod = new Module("sum.ll", *unwrap(LLVMGetGlobalContext()));
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
    AllocaInst *ptrA = new AllocaInst(IntegerType::get(mod->getContext(), 32), 0, "3", labelEntry);
    ptrA->setAlignment(MaybeAlign(4));
    AllocaInst *ptrB = new AllocaInst(IntegerType::get(mod->getContext(), 32), 0, "4", labelEntry);
    ptrB->setAlignment(MaybeAlign(4));

    StoreInst *st0 = new StoreInst(int32_0, ptrA, false, labelEntry);
    st0->setAlignment(MaybeAlign(4));
    StoreInst *st1 = new StoreInst(int32_1, ptrB, false, labelEntry);
    st1->setAlignment(MaybeAlign(4));
    LoadInst *ld0 = new LoadInst(ptrA, "", false, labelEntry);
    ld0->setAlignment(MaybeAlign(4));
    LoadInst *ld1 = new LoadInst(ptrB, "", false, labelEntry);
    ld1->setAlignment(MaybeAlign(4));

    BinaryOperator *addRes = BinaryOperator::Create(Instruction::Add, ld0, ld1, "add", labelEntry);
    ReturnInst::Create(mod->getContext(), addRes, labelEntry);

    return mod;
}

int main() {
    Module *Mod = makeLLVMModule();
    auto out_stream = new raw_os_ostream(cout);
    verifyModule(*Mod, out_stream);
    error_code ErrorInfo;
    auto Out = new ToolOutputFile("./sum.bc", ErrorInfo, sys::fs::F_None);
    
    cout << Mod->getTargetTriple() << endl;
    cout << Mod->getInstructionCount() << endl;
    WriteBitcodeToFile(*Mod, Out->os());
    Out->keep();

    return 0;
}
