#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

namespace {
    class SkeletonPass: public FunctionPass {
        public:
        static char ID;
        SkeletonPass() : FunctionPass(ID) {}

        virtual bool runOnFunction(Function &F) {
            errs() << "In a function called  " << F.getName() << "!\n";
            errs() << "Function body: \n";
            F.print(llvm::errs());

            for (auto &B : F) {
                errs() << "Basic block:\n";
                B.print(llvm::errs(), true);

                for (auto &I : B) {
                    errs() << "Instruction: \n";
                    I.print(llvm::errs(), true);
                    errs() << "\n";
                }
            }

            return false;
        }
    };
}

char SkeletonPass::ID = 0;
static void registerSkeletonPass(const PassManagerBuilder &, legacy::PassManagerBase &PM) {
    PM.add(new SkeletonPass());
};

static RegisterStandardPasses RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible, registerSkeletonPass);