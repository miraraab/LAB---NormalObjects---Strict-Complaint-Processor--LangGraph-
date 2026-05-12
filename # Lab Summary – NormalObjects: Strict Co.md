# Lab Summary – NormalObjects: Strict Complaint Processor (LangGraph)

LangGraph and LangChain represent two fundamentally different approaches to building AI workflows. 
LangChain agents operate in a freeform loop where the model decides at runtime which tools to call 
and in what order — this gives flexibility and creativity, but makes behavior harder to predict or audit. 
LangGraph, by contrast, enforces a strict state machine: each node has a defined role, transitions 
are explicit, and the full workflow path is traceable. For this complaint processing use case, 
LangGraph is the clear choice — compliance systems, customer service pipelines, and any process 
requiring auditability benefit from guaranteed step execution and consistent output structure. 
The trade-off is reduced flexibility: adding a new processing step requires explicit graph changes, 
whereas a LangChain agent might adapt on its own. Use LangChain when exploration and autonomy 
matter; use LangGraph when consistency, traceability, and control are non-negotiable.