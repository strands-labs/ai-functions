/-
The support executable: the warm session server and the cold checker.

Every mode reads JSON requests on stdin and writes one JSON reply per line on
stdout; a request that cannot be handled gets `{"fatal": str}`.

- no arguments: the session server. It holds one committed `Command.State`;
  every request elaborates against it, and only a successful `commit`
  replaces it. The server is not trusted: every certificate is compiled and
  checked cold.
- `compile`, `check`: one request each, in a fresh process (`Check.lean`).

Session ops (`model` is the byte ranges of `source` the model wrote, confined
by `Confine.lean`):
  {"op":"init","imports":[str],"source":str,"module":str?}
  {"op":"elab","source":str,"model":[[start,stop]]?,"commit":bool,"axioms":[str]?}
  {"op":"eval","source":str,"model":[[start,stop]]?,"names":[str]}
  {"op":"closed","source":str,"model":[[start,stop]]?,"name":str}
  {"op":"parse_term","source":str}
  {"op":"describe","names":[[spelling,[parts]]],"owned":[str]}
Elaboration replies:
  {"ok","messages":[{severity,line,col,offset,text,unknown}],"stdout","consts":[str],
   "declared":[[parts]],"rejected"?}, plus on success
  "axioms":{name:[str]} (elab), "values":[json|null],"types":[{type,type_str}] (eval),
  or "term":str (closed).
-/
import Lean
import Introspect
import Check

open Lean Elab Meta

def options : Options :=
  ({} : Options).setBool `Elab.async false

/-- Run `x` under `opts`: the committed scope's options, so the project's `set_option`s
bound reduction here as they do elaboration. -/
def runMeta (env : Environment) (x : MetaM α) (opts : Options := options) : IO α := do
  let ctx : Core.Context :=
    { fileName := "<ltx>", fileMap := default, options := opts, maxRecDepth := maxRecDepth.get opts }
  return (← x.run'.toIO ctx { env }).1

def get [FromJson α] (req : Json) (key : String) : IO α :=
  IO.ofExcept (req.getObjValAs? α key)

/-- A field that may be absent, but not malformed. -/
def getOr [FromJson α] [EmptyCollection α] (req : Json) (key : String) : IO α :=
  return (← get req key : Option α).getD {}

def localNames (env : Environment) : BaseIO NameSet := do
  return (← env.getLocalConstantInfos).foldl (·.insert ·.name) {}

def messageJson (fileMap : FileMap) (m : Message) : IO Json := do
  let severity := match m.severity with
    | .information => "info"
    | .warning => "warning"
    | .error => "error"
  return Json.mkObj
    [("severity", severity), ("line", m.pos.line), ("col", m.pos.column),
     ("offset", (fileMap.ofPosition m.pos).byteIdx), ("text", ← m.data.toString),
     ("unknown", m.data.hasTag (· == unknownIdentifierMessageTag))]

/-- A name as its components. -/
def partsJson (n : Name) : Json :=
  toJson (n.components.map fun | .str _ s => s | c => c.toString)

/-- Elaborate `source` against `st`, confining `model`. The caller decides
whether to keep the new state. -/
def process (st : Command.State) (source : String) (model : AIFunctionsConfine.Ranges) :
    IO (Command.State × Json) := do
  let before ← localNames st.env
  let ictx := Parser.mkInputContext source "<ltx>"
  let (output, result) ← IO.FS.withIsolatedStreams (isolateStderr := false) <|
    AIFunctionsConfine.run ictx model {} { st with messages := {}, infoState := {} }
  let new := { result.commandState with messages := {}, infoState := {} }
  let messages := result.commandState.messages.toArray
  let consts := (← localNames new.env).foldl (fun acc n => if before.contains n then acc else acc.push n) #[]
  -- Constants the block wrote itself: with a source range, not nested under another.
  let written := consts.filter fun n => !n.isInternal && (declRangeExt.find? new.env n).isSome
  let declared := written.filter fun n => !written.any fun m => m != n && m.isPrefixOf n
  let reply := Json.mkObj
    [("ok", !messages.any (·.severity matches .error)),
     ("messages", Json.arr (← messages.mapM (messageJson ictx.fileMap))), ("stdout", output),
     ("consts", toJson (consts.map toString)), ("declared", Json.arr (declared.map partsJson))]
  return (new, match result.rejected with
    | some text => reply.setObjVal! "rejected" text
    | none => reply)

/--
Reduce a closed boundary value to JSON, as Python's `lean.types.decode` reads it
(`Nat`/`Int` → number, `String` → string, `Bool` → bool, `List` → array, `Prod` →
`[fst, snd]`). Reduction is `whnf` at default transparency, so a term depending on
an `opaque` gets stuck and comes back `none`, unlike compiled `#eval`.
-/
partial def fragToJson (e : Expr) : MetaM (Option Json) := do
  let e ← whnf e
  match e with
  | .lit (.natVal n) => return some (toJson n)
  | .lit (.strVal s) => return some (toJson s)
  | _ =>
    let args := e.getAppArgs
    match e.getAppFn with
    | .const c _ =>
      if c == ``Bool.true then return some (toJson true)
      else if c == ``Bool.false then return some (toJson false)
      else if c == ``Nat.succ && args.size == 1 then
        match (← fragToJson args[0]!).bind (·.getNat?.toOption) with
        | some n => return some (toJson (n + 1))
        | none => return none
      else if c == ``Int.ofNat && args.size == 1 then
        fragToJson args[0]!
      else if c == ``Int.negSucc && args.size == 1 then
        match (← fragToJson args[0]!).bind (·.getNat?.toOption) with
        | some n => return some (toJson (-(n : Int) - 1))
        | none => return none
      else if c == ``List.nil then
        return some (Json.arr #[])
      else if c == ``List.cons && args.size == 3 then
        let some hd ← fragToJson args[1]! | return none
        match ← fragToJson args[2]! with
        | some (Json.arr tl) => return some (Json.arr (#[hd] ++ tl))
        | _ => return none
      else if c == ``Prod.mk && args.size == 4 then
        let some l ← fragToJson args[2]! | return none
        let some r ← fragToJson args[3]! | return none
        return some (Json.arr #[l, r])
      else return none
    | _ => return none

/-- The value of constant `n` reduced to fragment JSON, `null` when stuck or absent, and
`{"error": text}` when reduction fails, for example on a resource limit. -/
def evalConst (env : Environment) (opts : Options) (n : Name) : IO Json := do
  let some info := env.find? n | return .null
  let some value := info.value? | return .null
  if info matches .opaqueInfo _ then return .null
  try return (← runMeta env (fragToJson value) opts).getD .null
  catch e => return Json.mkObj [("error", toString e)]

def constType (env : Environment) (n : Name) : IO Json := do
  let some info := env.find? n | return .null
  runMeta env do
    return Json.mkObj [("type", ← aiFunctionsType info.type), ("type_str", (← ppExpr info.type).pretty)]

def errorReply (text : String) : Json :=
  Json.mkObj [("ok", false), ("messages", Json.arr #[Json.mkObj [("severity", "error"), ("text", text)]])]

/-- The value of `name` after elaborating `source` in `st`, with the constants the session
declared (those not in `base`) unfolded, printed with full names. Fail closed: the text must
re-elaborate over `base` alone, confined as model text, to a kernel-defeq value and type. -/
def closedTerm (base st : Command.State) (source : String) (model : AIFunctionsConfine.Ranges)
    (name : Name) : IO Json := do
  let (new, reply) ← process st source model
  if reply.getObjValD "ok" != true then return reply
  let some info := new.env.find? name | return errorReply s!"Unknown declaration: {name}"
  let some value := info.value? | return errorReply s!"{name} has no value"
  let declared (n : Name) := !base.env.contains n
  let (type, text) ← runMeta (opts := new.scopes.head!.opts) new.env <| withOptions (·.setBool `pp.fullNames true) do
    let print (e : Expr) : MetaM String := return (← ppExpr (← deltaExpand e declared)).pretty 1000000
    return (← print info.type, ← print value)
  let head := s!"def {name} : {type} :=\n"
  let (checked, again) ← process base (head ++ text) #[(head.utf8ByteSize, head.utf8ByteSize + text.utf8ByteSize)]
  let same := again.getObjValD "ok" == true && match checked.env.find? name with
    | some c => c.value?.any (Kernel.isDefEqGuarded new.env {} · value) &&
        Kernel.isDefEqGuarded new.env {} c.type info.type
    | none => false
  unless same do
    return errorReply s!"`{text}` does not re-elaborate over the project alone to the same term; \
      it may depend on an axiom of the ledger, such as a J judgment."
  return reply.setObjVal! "term" text

/-- Parse `source` as exactly one term: `leading_by`, and on failure where a
complete term ended before the input did. -/
def parseTerm (env : Environment) (source : String) : Json :=
  let ictx := Parser.mkInputContext source "<input>"
  let parse p := p.run ictx { env, options } (Parser.getTokenTable env) (Parser.mkParserState source)
  let leadingBy := !(parse (Parser.andthenFn Parser.whitespace (Parser.symbolFn "by"))).hasError
  let s := parse (Parser.andthenFn Parser.whitespace (Parser.categoryParserFnImpl `term))
  let failure (s : Parser.ParserState) (ended : Json) := Json.mkObj
    [("ok", false), ("leading_by", leadingBy), ("ended", ended),
     ("messages", Json.arr #[Json.mkObj [("severity", "error"), ("text", s.toErrorMsg ictx)]])]
  if !s.allErrors.isEmpty then failure s .null
  else if ictx.atEnd s.pos then Json.mkObj [("ok", true), ("leading_by", leadingBy)]
  else
    let pos := ictx.fileMap.toPosition s.pos
    failure (s.mkError "end of input") (Json.mkObj [("line", pos.line), ("col", pos.column)])

def session (ref base : IO.Ref (Option Command.State)) (req : Json) : IO Json := do
  let op : String ← get req "op"
  if op == "init" then
    let imports : Array String ← get req "imports"
    let module : Option String ← get req "module"
    let env ← importModules (imports.map ({ module := ·.toName })) options (leakEnv := true) (loadExts := true)
    let env := env.setMainModule ((module.map String.toName).getD `LtxSession)
    let (st, reply) ← process (Command.mkState env {} options) (← get req "source") #[]
    if reply.getObjValD "ok" == true then
      ref.set st
      base.set st
    return reply
  let some st ← ref.get | throw (IO.userError s!"{op} before init")
  match op with
  | "elab" =>
    let (new, reply) ← process st (← get req "source") (← getOr req "model")
    if reply.getObjValD "ok" != true then return reply
    let inventories ← (← getOr req "axioms" : Array String).mapM fun s => do
      unless new.env.contains s.toName do throw (IO.userError s!"Unknown declaration: {s}")
      return (s, toJson ((← runMeta new.env (collectAxioms s.toName)).map toString))
    if (← get req "commit" : Bool) then ref.set new
    return reply.setObjVal! "axioms" (Json.mkObj inventories.toList)
  | "eval" =>
    let (new, reply) ← process st (← get req "source") (← getOr req "model")
    if reply.getObjValD "ok" != true then return reply
    let names : Array String ← get req "names"
    let values ← names.mapM (evalConst new.env new.scopes.head!.opts ·.toName)
    let types ← names.mapM (constType new.env ·.toName)
    return (reply.setObjVal! "values" (Json.arr values)).setObjVal! "types" (Json.arr types)
  | "closed" =>
    let some b ← base.get | throw (IO.userError "closed before init")
    closedTerm b st (← get req "source") (← getOr req "model") (← get req "name" : String).toName
  | "parse_term" => return parseTerm st.env (← get req "source")
  | "describe" =>
    let names : Array (String × Array String) ← get req "names"
    let owned : Array String ← get req "owned"
    let mut symbols := #[]
    for (spelling, parts) in names do
      let name := AIFunctionsCheck.nameOf parts
      let some info ← runMeta st.env (aiFunctionsDescribe st.env (owned.map String.toName) spelling name)
        | return Json.mkObj [("ok", false), ("error", s!"Unknown declaration: {spelling}")]
      symbols := symbols.push info
    return Json.mkObj [("ok", true), ("symbols", Json.arr symbols)]
  | other => throw (IO.userError s!"unknown op `{other}`")

/-- Answer requests until stdin closes. -/
partial def serve (handle : Json → IO Json) : IO Unit := do
  let line ← (← IO.getStdin).getLine
  if line.isEmpty then return
  unless line.trimAscii.isEmpty do
    let reply ← try handle (← IO.ofExcept (Json.parse line)) catch e => pure (Json.mkObj [("fatal", toString e)])
    let stdout ← IO.getStdout
    stdout.putStr (reply.compress ++ "\n")
    stdout.flush
  serve handle

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  match args with
  | [] =>
    enableInitializersExecution
    serve (session (← IO.mkRef none) (← IO.mkRef none))
  | ["compile"] => serve AIFunctionsCheck.compile
  | ["check"] => serve AIFunctionsCheck.check
  | _ => IO.eprintln "usage: server [compile | check]"; return 2
  return 0
