import type { PhraseDecision } from "../types";

interface Props {
  decision: PhraseDecision;
  busy: boolean;
  onChoose: (phrase: string, attributeId: string) => void;
}

export function ClarificationCard({ decision, busy, onChoose }: Props) {
  const outcomeLabel =
    decision.outcome === "none" ? "No governed match" : `${decision.outcome} match — needs confirmation`;

  return (
    <div className={`clarify-card clarify-card--${decision.outcome}`}>
      <div className="clarify-card__head">
        <span className="clarify-card__phrase">"{decision.phrase}"</span>
        <span className="clarify-card__outcome">{outcomeLabel}</span>
      </div>
      <p>{decision.clarification_message ?? "No governed attribute matched this phrase closely enough."}</p>
      {decision.options.length > 0 && (
        <div className="clarify-card__options">
          {decision.options.map((opt) => (
            <button
              key={opt.attribute.attribute_id}
              className="btn btn--option"
              disabled={busy}
              onClick={() => onChoose(decision.phrase, opt.attribute.attribute_id)}
            >
              {opt.attribute.business_name}
              <code>{opt.attribute.attribute_id}</code>
              {opt.attribute.governance?.pii && <span className="badge badge--pii">PII</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
