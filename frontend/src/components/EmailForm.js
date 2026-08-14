import React, { useState } from 'react';

const SAMPLE_EMAILS = [
  {
    label: "Billing Issue",
    text: "Hi support, I was billed twice for my annual subscription renewal yesterday. Can you please process a refund for $99?"
  },
  {
    label: "Technical Bug",
    text: "I am getting a 500 Server Error whenever I click the Export PDF button on my dashboard analytics tab."
  },
  {
    label: "Account Access",
    text: "I am unable to receive password reset emails and cannot log into my workspace account."
  },
  {
    label: "General Question",
    text: "What are your business hours and do you offer custom enterprise pricing plans?"
  }
];

export default function EmailForm({ onSubmit, loading }) {
  const [emailText, setEmailText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!emailText.trim() || loading) return;
    onSubmit(emailText.trim());
  };

  const handleQuickFill = (text) => {
    setEmailText(text);
  };

  return (
    <div className="glass-card rounded-2xl p-6 shadow-xl border border-slate-800">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>✉️</span> Real-Time Email Classifier
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Input customer support email text below to classify into Billing, Technical, Account, or General.
          </p>
        </div>
        <span className="text-xs text-slate-500 font-mono">
          {emailText.length} chars
        </span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <textarea
            rows="5"
            className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all resize-none"
            placeholder="Type or paste customer support email here... e.g. 'I was charged twice for my subscription this month.'"
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
            disabled={loading}
          />
        </div>

        {/* Quick Sample Fill Buttons */}
        <div>
          <span className="text-xs text-slate-400 font-medium block mb-2">Try sample templates:</span>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_EMAILS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleQuickFill(sample.text)}
                disabled={loading}
                className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-all"
              >
                + {sample.label}
              </button>
            ))}
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-2 flex justify-end">
          <button
            type="submit"
            disabled={!emailText.trim() || loading}
            className={`px-6 py-3 rounded-xl font-semibold text-sm shadow-lg flex items-center gap-2 transition-all ${
              !emailText.trim() || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30 hover:shadow-indigo-500/50 active:scale-95'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing AI Model...</span>
              </>
            ) : (
              <>
                <span>Classify Email</span>
                <span>🚀</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
