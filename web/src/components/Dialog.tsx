// Promise-based in-app dialogs (replace window.prompt/confirm, which block
// the page, cannot be styled, and are suppressed in some embedded browsers).
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

type Request =
  | { kind: "prompt"; title: string; label?: string; initial: string; confirm: string; resolve: (value: string | null) => void }
  | { kind: "confirm"; title: string; body?: string; confirm: string; danger?: boolean; resolve: (value: boolean) => void };

interface DialogApi {
  prompt: (title: string, initial?: string, options?: { label?: string; confirm?: string }) => Promise<string | null>;
  confirm: (title: string, options?: { body?: string; confirm?: string; danger?: boolean }) => Promise<boolean>;
}

const DialogContext = createContext<DialogApi | null>(null);

export function useDialog(): DialogApi {
  const value = useContext(DialogContext);
  if (!value) throw new Error("useDialog outside DialogProvider");
  return value;
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<Request | null>(null);
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);

  const prompt = useCallback<DialogApi["prompt"]>((title, initial = "", options = {}) => new Promise((resolve) => {
    setText(initial);
    setRequest({ kind: "prompt", title, label: options.label, initial, confirm: options.confirm ?? "OK", resolve });
  }), []);
  const confirm = useCallback<DialogApi["confirm"]>((title, options = {}) => new Promise((resolve) => {
    setRequest({ kind: "confirm", title, body: options.body, confirm: options.confirm ?? "OK", danger: options.danger, resolve });
  }), []);

  useEffect(() => {
    if (request?.kind === "prompt") setTimeout(() => input.current?.select(), 0);
  }, [request]);

  const close = (accepted: boolean) => {
    if (!request) return;
    if (request.kind === "prompt") request.resolve(accepted && text.trim() ? text.trim() : null);
    else request.resolve(accepted);
    setRequest(null);
  };

  return (
    <DialogContext.Provider value={{ prompt, confirm }}>
      {children}
      {request && (
        <div className="modal-backdrop" onMouseDown={() => close(false)}>
          <div className="modal small" role="dialog" aria-modal="true" aria-label={request.title} onMouseDown={(e) => e.stopPropagation()}
            onKeyDown={(e) => { if (e.key === "Escape") close(false); if (e.key === "Enter") close(true); }}>
            <h3>{request.title}</h3>
            {request.kind === "prompt" && (
              <label className="dialog-field">{request.label}
                <input ref={input} value={text} onChange={(e) => setText(e.target.value)} />
              </label>
            )}
            {request.kind === "confirm" && request.body && <p className="muted">{request.body}</p>}
            <div className="button-row end">
              <button onClick={() => close(false)}>Cancel</button>
              <button className={request.kind === "confirm" && request.danger ? "danger" : "primary"} autoFocus={request.kind === "confirm"}
                disabled={request.kind === "prompt" && !text.trim()} onClick={() => close(true)}>{request.confirm}</button>
            </div>
          </div>
        </div>
      )}
    </DialogContext.Provider>
  );
}
