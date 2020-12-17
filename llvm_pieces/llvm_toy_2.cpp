#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Value.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace llvm;

/* The lexer*/
enum Token {
    tok_eof = -1, 

    // commonds
    tok_def = -2,
    tok_extern = -3,

    // primary
    tok_identifier = -4,
    tok_number = -5 
};

static string IdentifierStr; // Filled in if tok_identifier
static double NumVal; //Filled in if tok_number

// gettok - Return the next token from stantdard input.  
static int gettok() {
    static int LastChar = ' ';
    
    // skip the whitespace
    while (isspace(LastChar)) {
        LastChar = getchar();
    }

    // identifier: [a-zA-Z][a-zA-Z0-9]*   
    if (isalpha(LastChar)) { 
        IdentifierStr = LastChar;
        while(isalnum((LastChar = getchar()))) {
            IdentifierStr += LastChar;
        }

        if (IdentifierStr == "def") {
            return tok_def;
        }
        if (IdentifierStr == "extern") {
            return tok_extern;
        }

        return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(LastChar) || LastChar == '.') {
        string NumStr;
        do {
            NumStr += LastChar;
            LastChar = getchar();
        }while(isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), nullptr);
        return tok_number;
    }

    if (LastChar == '#') {
        // Comment until the end of the line
        do {
            LastChar = getchar();
        } while(LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF) {
            return gettok();
        }
    }

    // Check the end of the file
    if (LastChar == EOF) {
        return tok_eof;
    }

    // Otherwise just return the character as the ascii value
    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}

/* Abstraction Syntax Tree --- AST, the parse tree*/
namespace {

// ExprAST, base class for all expression nodes
class ExprAST {
    public:
    virtual ~ExprAST() = default;
    virtual Value *codegen() = 0;
};

// NumberExprAST - Expression class for number like '1.0'
class NumberExprAST: public ExprAST {
    double Val;

    public:
    NumberExprAST(double val):Val(val) {}
    Value *codegen() override;
};

// VariableExprAST - Expression class for referecing a varible
class VariableExprAST: public ExprAST {
    string Name;

    public:
    VariableExprAST(const string &name): Name(name) {}
    Value *codegen() override;
};

// BinaryExprAST - Expression class for binary operator
class BinaryExprAST: public ExprAST {
    char Op;
    unique_ptr<ExprAST> LHS, RHS;

    public:
    BinaryExprAST(char Op, unique_ptr<ExprAST> LHS, unique_ptr<ExprAST> RHS):
        Op(Op), LHS(move(LHS)), RHS(move(RHS)) {}
    Value *codegen() override;
};

// CallExprAST - Expression class for function calls
class CallExprAST: public ExprAST {
    string Callee;
    vector<unique_ptr<ExprAST>> Args;

    public:
    CallExprAST(const string &Callee, vector<unique_ptr<ExprAST>> Args): 
        Callee(Callee), Args(move(Args)) {}
    Value *codegen() override;
};

// PrototypeAST -- This class represents the "prototype" for a function,
// which captures its name, and its argument names (thus implicitly the number
// of arguments the function takes).
class PrototypeAST {
    string Name;
    vector<string> Args;

    public:
    PrototypeAST(const string &Name, vector<string> Args):
        Name(Name), Args(move(Args)) {}

    const string &getName() const { return Name; }
    Function *codegen();
};

// FunctionAST --- reprents a function definition
class FunctionAST {
    unique_ptr<PrototypeAST> Proto;
    unique_ptr<ExprAST> Body;

    public: 
    FunctionAST(unique_ptr<PrototypeAST> Proto, unique_ptr<ExprAST> Body):
        Proto(move(Proto)), Body(move(Body)) {}
    Function *codegen();
}; 

} // end of namespace


/* Parser */
/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

// BinopPrecedence - the precedence for each binary operator token
static map<char, int> BinopPrecedence;

// GetTokPrecedence - get the precedence of the token from map
static int GetTokPrecedence() {
    if (!isascii(CurTok)) {
        return -1;
    }

    int TokPrec = BinopPrecedence[CurTok];
    if (TokPrec <= 0) {
        return -1;
    }
    return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
unique_ptr<ExprAST> LogError(const char *Str) {
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}
unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
    LogError(Str);
    return nullptr;
}

static unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static unique_ptr<ExprAST> ParseNumberExpr() {
    auto res = make_unique<NumberExprAST>(NumVal);
    getNextToken();
    return move(res);
}

/// parenexpr ::= '(' expression ')'
static unique_ptr<ExprAST> ParseParenExpr() {
    getNextToken(); // eat (
    auto v = ParseExpression();

    if (!v) {
        return nullptr;
    }

    if (CurTok != ')') {
        return LogError("expected ')'");
    }

    getNextToken(); // eat )
    return v;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static unique_ptr<ExprAST> ParseIdentifierExpr() {
    string IdName = IdentifierStr;

    getNextToken(); // eat IdentifierStr
    if (CurTok != '(') {
        return make_unique<VariableExprAST>(IdName);
    }

    // Call
    getNextToken(); // eat  (
    vector<unique_ptr<ExprAST>> Args;
    if (CurTok != ')') {
        while(true) {
            if (auto Arg = ParseExpression()) {
                Args.push_back(move(Arg));
            }
            else {
                return nullptr;
            }

            if (CurTok == ')') {
                break;
            }

            if (CurTok != ',') {
                return LogError("Expected ')' or ',' in argument list");
            }

            getNextToken(); 
        }
    }

    getNextToken(); // eat ')'

    return make_unique<CallExprAST>(IdName, move(Args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static unique_ptr<ExprAST> ParsePrimary() {
    switch(CurTok) {
        default: 
            return LogError("unknown token when expecting an expression");
        case tok_identifier:
            return ParseIdentifierExpr();
        case tok_number:
            return ParseNumberExpr();
        case '(':
            return ParseParenExpr();
    }
}

/// binoprhs
///   ::= ('+' primary)*
static unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, unique_ptr<ExprAST> LHS) {
    //if binop, find precedence
    while(1) {
        int TokPrec = GetTokPrecedence();

        if (TokPrec < ExprPrec) {
            return LHS;
        }

        int BinOp = CurTok;
        getNextToken(); // eat the binop

        // Parse the primary expression after binop
        auto RHS = ParsePrimary();
        if (!RHS) {
            return nullptr;
        }

        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec) {
            RHS = ParseBinOpRHS(TokPrec + 1, move(RHS));
            if (!RHS) {
                return nullptr;
            }
        }

        // Merge LHS, RHS
        LHS = make_unique<BinaryExprAST>(BinOp, move(LHS), move(RHS));

    }
}

/// expression
///   ::= primary binoprhs
///
static unique_ptr<ExprAST> ParseExpression() {
    auto LHS = ParsePrimary();
    if (!LHS) {
        return nullptr;
    }

    return ParseBinOpRHS(0, move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
unique_ptr<PrototypeAST> ParsePrototype() {
    if (CurTok != tok_identifier) {
        return LogErrorP("Expected function name in prototype");
    }

    string FnName = IdentifierStr;
    getNextToken();

    if (CurTok != '(') {
        return LogErrorP("Expected ( in prototype");
    }

    vector<string> ArgNames;
    while(getNextToken() == tok_identifier) {
        ArgNames.push_back(IdentifierStr);
    }

    if (CurTok != ')') {
        return LogErrorP("Expected ) in prototype");
    }

    getNextToken(); //eat )

    return make_unique<PrototypeAST>(FnName, move(ArgNames));
}

/// definition ::= 'def' prototype expression
static unique_ptr<FunctionAST> ParseDefinition() {
    getNextToken(); // eat def
    auto Proto = ParsePrototype();
    if (!Proto) {
        return nullptr;
    }

    if (auto E = ParseExpression()) {
        return make_unique<FunctionAST>(move(Proto), move(E));
    }

    return nullptr;
}

/// toplevelexpr ::= expression
static unique_ptr<FunctionAST> ParseTopLevelExpr() {
    if (auto E = ParseExpression()) {
        auto Proto = make_unique<PrototypeAST>("__anon_expr", vector<string>());
        
        return make_unique<FunctionAST>(move(Proto), move(E));
    }

    return nullptr;
}

/// external ::= 'extern' prototype
static unique_ptr<PrototypeAST> ParseExtern() {
    getNextToken(); // eat extern

    return ParsePrototype();
}

/*  Code generator */
static unique_ptr<LLVMContext> TheContext;
static unique_ptr<Module> TheModule;
static unique_ptr<IRBuilder<>> Builder;
static map<string, Value* > NamedValues;

Value *LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}

Value *NumberExprAST::codegen() {
    return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
    Value *V = NamedValues[Name];
    if (!V) {
        return LogErrorV("Unkown variable name");
    }

    return V;
}

Value *BinaryExprAST::codegen() {
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();

    if (!L || !R) {
        return nullptr;
    }

    switch(Op) {
        case '+': 
            return Builder->CreateFAdd(L, R, "addtmp");
        case '-':
            return Builder->CreateFSub(L, R, "subtmp");
        case '*':
            return Builder->CreateFMul(L, R, "multmp");
        case '<':
            L = Builder->CreateFCmpULT(L, R, "cmptmp");
            return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
        default:
            return LogErrorV("invalid binary operator");
    }
}

Value *CallExprAST::codegen() {
    Function *CalleeF = TheModule->getFunction(Callee);
    if (!CalleeF) {
        return LogErrorV("Unkown function referenced");
    }

    if (CalleeF->arg_size() != Args.size()) {
        return LogErrorV("Incorrect # arguments passed");
    }

    vector<Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; i++) {
        ArgsV.push_back(Args[i]->codegen());
        if (!ArgsV.back()) {
            return nullptr;
        }
    }

    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Function *PrototypeAST::codegen() {
    vector<Type*> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
    FunctionType *FT = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    unsigned idx = 0;
    for (auto &Arg : F->args()) {
        Arg.setName(Args[idx++]);
    }

    return F;
}

Function *FunctionAST::codegen() {
    Function *TheFunction = TheModule->getFunction(Proto->getName());

    if (!TheFunction) {
        TheFunction = Proto->codegen();
    }

    if (!TheFunction) {
        return nullptr;
    }

    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    NamedValues.clear();
    for (auto &Arg : TheFunction->args()) {
        NamedValues[string(Arg.getName())] = &Arg;
    }

    if (Value *RetVal = Body->codegen()) {
        Builder->CreateRet(RetVal);
        verifyFunction(*TheFunction);
        return TheFunction;
    }

    TheFunction->eraseFromParent();
    return nullptr;
}

/* Top level parser and JIT Driver*/
static void InitializeModule() {
    TheContext = make_unique<LLVMContext>();
    TheModule = make_unique<Module>("my cool jit", *TheContext);

    Builder = make_unique<IRBuilder<>>(*TheContext);
}

static void HandleDefinition() {
    if (auto FnAST = ParseDefinition()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "read fucntion definition");
            FnIR->print(errs());
            fprintf(stderr, "\n");
        }
    } else {
        getNextToken();
    }
}

static void HandleExtern() {
    if (auto ProtoAST = ParseExtern()) {
        if (auto *FnIR = ProtoAST->codegen()) {
            fprintf(stderr, "Read extern: ");
            FnIR->print(errs());
            fprintf(stderr, "\n");
        }
    } else {
        getNextToken();
    }
}

static void HandleTopLevelExression() {
    if (auto FnAST = ParseTopLevelExpr()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read top-level expression: ");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            FnIR->eraseFromParent();
        }
    } else {
        getNextToken();
    }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
    while(true) {
        fprintf(stderr, "ready> ");
        switch (CurTok) {
            case tok_eof:
                return;
            case  ';':
                getNextToken();
                break;
            case tok_def:
                HandleDefinition();
                break;
            case tok_extern:
                HandleExtern();
                break;
            default:
                HandleTopLevelExression();
                break;
        }
    }
}

int main() {

    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40;

     // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();

    // Make the module, which holds all the codes
    InitializeModule();

    // Run the main "interpreter loop" now.
    MainLoop();

    // Print out all of the generated codes
    TheModule->print(errs(), nullptr);

    return 0;
}