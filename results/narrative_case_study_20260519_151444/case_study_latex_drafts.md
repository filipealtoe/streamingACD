# LaTeX Drafts for §4 Case Study

Date: 2026-05-19
Source: `narrative_candidates.md` in this directory.

All drafts assume the dimensional check-worthiness scores will be filled in
once the four-head checkpoint is run over the 535 normalized claims.
Placeholders are written as `\todo{checkability=...}` etc.; remove the
`\todo{...}` macro and substitute numbers before submission. The drafts use
the `viral_confidence` proxy for the overall confidence; if the paper wants
the actual four-head overall instead, swap it in at the same location.

All drafts are em-dash free.

---

## Candidate 1 (recommended): mail-in ballot counting in battleground states

### Compact variant

```latex
We illustrate the pipeline with a representative narrative from the 2020 US
election corpus. On 2020-11-04 at 07:00 UTC, the streaming clusterer flagged
cluster 23761 as anomalous with a z-score of 3.37 over the cumulative
tweet-count signal. At that point the cluster had accumulated 38 units of
engagement; over the next five hours engagement rose to 3{,}479 at the cluster's
peak, and the final post-event total was 5{,}416. The normalized claim
emitted by the claim-extraction module was
\textit{``Counting mail-in ballots in key battleground states will skew the
election results in favor of Trump.''} The four parallel check-worthiness
heads returned \todo{checkability=$c_1$, verifiability=$c_2$, harm=$c_3$},
with an overall recommendation confidence of $0.85$, above our
operating threshold of $0.7$. The cluster's measured Preventable Spread
Ratio was $0.99$, confirming that the early flag corresponded to a
narrative that subsequently dominated the post-event engagement window.
\end{quote}
```

(Remove `\end{quote}` if the surrounding context is not a `quote` env.)

### Boxed variant

```latex
\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{
\small
\textbf{Case study: cluster 23761, ``mail-in counting in battleground states.''}

\smallskip
\textbf{Normalized claim:} ``Counting mail-in ballots in key battleground
states will skew the election results in favor of Trump.''

\smallskip
\textbf{Cluster contents (24 posts; representative paraphrases of three
members ordered by distance from the cluster centroid):}
\begin{itemize}
  \item A defense of slow state-by-state vote counting as the expected
        behavior of the United States electoral system, contrasted with
        framing it as irregular.
  \item A user expressing confusion about why all ballots are not simply
        counted before a winner is declared.
  \item A summary of asymmetric counting demands from one campaign:
        halt counting where the lead was eroding, continue where the lead
        might be regained.
\end{itemize}

\smallskip
\textbf{Pipeline trace:}
\begin{tabular}{ll}
Detection trigger & 2020-11-04 07:00 UTC, $z=3.37$ \\
Lead before peak  & 5.0 h \\
Engagement at detect / peak / final & 38 / 3{,}479 / 5{,}416 \\
4-head dimensional scores & \todo{$c_\mathrm{ck}, c_\mathrm{vf}, c_\mathrm{hp}$} \\
Overall confidence & $0.85$ \\
Preventable spread ratio & $0.99$ \\
\end{tabular}
}}
\caption{End-to-end pipeline trace for a representative procedural narrative
from the 2020 US election corpus. All quoted post content is paraphrased.}
\label{fig:case-study}
\end{figure}
```

---

## Candidate 2: Kentucky early vote counts

### Compact variant

```latex
A second narrative shows the pipeline's behavior on a longer-horizon cluster.
On 2020-11-03 at 23:00 UTC, the clusterer flagged cluster 25588 with a
z-score of 4.29; engagement at detection was 77, and the cluster's peak
engagement of 322 was reached only 92 hours later, on 2020-11-07.
The normalized claim was \textit{``Joe Biden is unexpectedly leading in
early vote counts in Kentucky, potentially challenging Donald Trump's
expected win.''} The four heads returned \todo{checkability=$c_1$,
verifiability=$c_2$, harm=$c_3$} with an overall confidence of $0.85$.
The measured Preventable Spread Ratio was $0.89$. The unusually long lead
illustrates that not every detection corresponds to a tightly clustered
engagement burst.
```

### Boxed variant

```latex
\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{
\small
\textbf{Case study: cluster 25588, ``Kentucky early vote counts.''}

\smallskip
\textbf{Normalized claim:} ``Joe Biden is unexpectedly leading in early vote
counts in Kentucky, potentially challenging Donald Trump's expected win.''

\smallskip
\textbf{Cluster contents (91 posts; representative paraphrases):}
\begin{itemize}
  \item An election-night observation that one candidate appeared to be
        winning Kentucky after roughly two-thirds of votes had been tallied.
  \item A pre-election poll reference comparing the projected Kentucky
        margin against the prior cycle's outcome.
  \item A commentary speculating whether the tighter-than-expected margin
        signaled broader weakness for the incumbent.
\end{itemize}

\smallskip
\textbf{Pipeline trace:}
\begin{tabular}{ll}
Detection trigger & 2020-11-03 23:00 UTC, $z=4.29$ \\
Lead before peak  & 92.0 h \\
Engagement at detect / peak / final & 77 / 322 / 676 \\
4-head dimensional scores & \todo{$c_\mathrm{ck}, c_\mathrm{vf}, c_\mathrm{hp}$} \\
Overall confidence & $0.85$ \\
Preventable spread ratio & $0.89$ \\
\end{tabular}
}}
\caption{A long-lead narrative whose peak engagement materialized days
after detection. Post content paraphrased.}
\label{fig:case-study-kentucky}
\end{figure}
```

---

## Candidate 3: Pennsylvania uncounted mail-in ballots

### Compact variant

```latex
A third case shows the pipeline behavior on a procedural narrative that
dominated the post-election news cycle. On 2020-11-04 at 07:00 UTC, the
clusterer flagged cluster 72106 with a z-score of 3.37; engagement at
detection was 1{,}031 and the cluster reached its peak of 2{,}678 fourteen
hours later, with a final post-event total of 6{,}851. The normalized claim
was \textit{``The legitimacy of Donald Trump's Pennsylvania win in the 2020
US presidential election is disputed due to uncounted mail-in ballots.''}
The four heads returned \todo{checkability=$c_1$, verifiability=$c_2$,
harm=$c_3$} with an overall confidence of $0.85$. The measured Preventable
Spread Ratio was $0.85$. The cluster contents are dominated by
county-level claims about remaining uncounted ballots, comparisons across
jurisdictions, and disputes over the counting timeline, illustrating the
procedural focus the pipeline is intended to surface.
```

### Boxed variant

```latex
\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{
\small
\textbf{Case study: cluster 72106, ``Pennsylvania uncounted ballots.''}

\smallskip
\textbf{Normalized claim:} ``The legitimacy of Donald Trump's Pennsylvania
win in the 2020 US presidential election is disputed due to uncounted
mail-in ballots.''

\smallskip
\textbf{Cluster contents (88 posts; representative paraphrases):}
\begin{itemize}
  \item A campaign-source claim cataloguing specific Pennsylvania counties
        with tens of thousands of uncounted ballots said to favor that
        campaign.
  \item A characterization of remarks predicting one candidate would lose
        Pennsylvania if all ballots were counted.
  \item A comparison invoking Taiwan's same-day vote-counting timeline to
        question why the Pennsylvania count was taking longer.
\end{itemize}

\smallskip
\textbf{Pipeline trace:}
\begin{tabular}{ll}
Detection trigger & 2020-11-04 07:00 UTC, $z=3.37$ \\
Lead before peak  & 14.0 h \\
Engagement at detect / peak / final & 1{,}031 / 2{,}678 / 6{,}851 \\
4-head dimensional scores & \todo{$c_\mathrm{ck}, c_\mathrm{vf}, c_\mathrm{hp}$} \\
Overall confidence & $0.85$ \\
Preventable spread ratio & $0.85$ \\
\end{tabular}
}}
\caption{A procedural post-election narrative around the Pennsylvania count.
Post content paraphrased.}
\label{fig:case-study-pa}
\end{figure}
```

---

## Notes for the human author

1. The "overall confidence" of $0.85$ in every draft is the pipeline's
   `viral_confidence` output, used as a proxy. If the four-head model is
   re-run and produces a different overall, prefer that.
2. The `\todo{...}` placeholders should be replaced before submission. They
   currently mark the three missing dimensional scores per candidate.
3. The clauses "claim was \textit{...}" quote the normalized claim
   verbatim. If the normalized claim wording is too partisan for §4 (as is
   the case for Candidate 1 and Candidate 3), reword the candidate-attack
   framing while preserving the procedural noun phrase.
4. None of the drafts mention user handles, hashtags, or message IDs. The
   `\todo` macro is from the `todonotes` package; replace with plain text if
   that package is not loaded.
