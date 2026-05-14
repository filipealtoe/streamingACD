# Pedido Para Filipe: Reprodutibilidade Da Avaliação Formativa

Filipe, há outro ponto da auditoria de reprodutibilidade que provavelmente depende de ti: os resultados quantitativos da
avaliação formativa com fact-checkers. Acho que estes dados podem estar no teu Google Forms ou num export local teu.

## O Que O Paper Diz

A versão IJCAI/paper menciona uma avaliação formativa com profissionais e reporta números como:

- `N=7` no abstract/limitations, mas `N=9` no corpo do paper;
- `27` claim-report pairs avaliados;
- `22/27` acordo ou acordo parcial;
- `16/27` acordo total;
- médias de utilidade `4.04`, `3.81`, `3.78`, `3.11`;
- `8/9` participantes alinhados com a estrutura do report;
- `7/9` indicando redução de esforço de triage;
- `5/9` indicando calibração/confiança.

## O Que Encontrámos No Meu Lado

Encontrámos os artefatos de preparação do estudo, mas não as respostas dos participantes:

```text
reproducibility/source_artifacts/formative_evaluation/select_claims_for_study.py
reproducibility/source_artifacts/formative_evaluation/expose_fast_study_claims_selection.csv
reproducibility/source_artifacts/formative_evaluation/expose_3day_full_study_claims_selection.csv
```

Estes ficheiros mostram que havia claims/reports selecionados para avaliação, mas **não chegam para reproduzir os números do
paper**.

O que falta:

- export do Google Forms;
- respostas anonimizadas por participante;
- mapping entre participante, claim/report e pergunta;
- script/notebook que calcula os agregados;
- decisão final sobre `N=7` vs `N=9`.

## O Que Precisamos Que Verifiques No Google Forms

Consegues procurar/exportar:

1. o formulário original;
2. o CSV/Sheets de respostas;
3. a lista de participantes, anonimizada;
4. o mapping dos `27` claim-report pairs usados no formulário;
5. as perguntas/escalas exatas;
6. qualquer notebook/script usado para calcular os valores do paper;
7. a versão final correta do `N`: `7` ou `9`.

Critério mínimo para manter os números no paper:

```text
responses export + mapping dos reports + analysis script -> 22/27, 16/27, médias, 8/9, 7/9, 5/9
```

## Formato Ideal Para Enviar

Idealmente, manda uma pasta com:

```text
formative_evaluation/
  responses_anonymized.csv
  report_pair_mapping.csv
  survey_questions.md
  analyze_formative_evaluation.py
  summary.json
```

O `summary.json` deveria conter algo deste género:

```json
{
  "n_participants": 9,
  "n_report_pairs": 27,
  "agreement_or_partial": "22/27",
  "full_agreement": "16/27",
  "usefulness_means": {
    "verifiability": 4.04,
    "checkability": 3.81,
    "overall_reasoning": 3.78,
    "virality": 3.11
  },
  "structure_alignment": "8/9",
  "triage_effort_reduction": "7/9",
  "trust_calibration": "5/9"
}
```

## Searches Úteis

No Google Drive / local:

```text
formative evaluation
fact-checker evaluation
expert review
study claims
claim report pairs
Google Forms
22/27
16/27
4.04
3.81
3.78
3.11
```

No repo/local:

```bash
rg -n '22/27|16/27|4\\.04|3\\.81|3\\.78|3\\.11|8/9|7/9|5/9|formative|fact-checker|Google Forms'
find . -iname '*formative*' -o -iname '*survey*' -o -iname '*responses*' -o -iname '*evaluation*'
```

## Decisão Se Não Encontrarmos

Se não encontrarmos o export das respostas e o script de análise, a opção segura é remover todos os números da avaliação
formativa do abstract/body e tratar isto apenas como protocolo/stimuli preparados.

Texto fallback possível:

```text
We prepared an expert-review protocol and selected report stimuli spanning high-confidence positive, negative recommendation, and
borderline cases. The current reproducibility package includes the selection script and selected report tables, but does not report
quantitative participant-response statistics.
```

Sem os dados do Google Forms, não devemos manter `22/27`, `16/27`, as médias, nem os contadores `8/9`, `7/9`, `5/9`.
