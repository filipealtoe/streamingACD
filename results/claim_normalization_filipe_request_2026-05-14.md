# Pedido Para Filipe: Reprodutibilidade Do Claim Normalization

Filipe, estamos a fechar a auditoria de reprodutibilidade do paper e há um ponto ainda bloqueado: os resultados de **claim
normalization** da Table 1.

## O Que O Paper Diz

A versão atual do paper reporta, para CheckThat! 2025 claim normalization:

- test set com `N=300`;
- valores METEOR da Table 1, incluindo `0.5583`, `0.5463` e `0.5691`;
- uma claim de melhoria de cerca de `10% METEOR` sobre SOTA.

## O Que Encontrámos No Meu Lado

No `explainableACD` local e no pacote que migrámos para `streamingACD`, o artefato mais próximo é:

```text
reproducibility/source_artifacts/claim_normalization/comparison_test_20260113_123010.json
```

Esse ficheiro **não reproduz a Table 1 do paper**:

- tem `n=1285`, não `N=300`;
- o melhor METEOR médio é `0.3449` (`deepseek-v3`);
- não contém as predições/sumário exatos que produzam `0.5583`, `0.5463` ou `0.5691`.

Também existem scripts e outputs de claim normalization em:

```text
experiments/scripts/run_claim_normalization_ct25.py
experiments/results/claim_normalization/
claim_norm/scripts/run_claim_normalization_ct25.py
claim_norm/prompts/claim_normalization.yaml
```

Mas, até agora, não encontrei o pacote exato da Table 1.

## O Que Precisamos Que Verifiques No Teu PC

Consegues procurar se tens a execução original da Table 1? Precisamos de qualquer artefato que prove o resultado com `N=300`.

Idealmente, envia ou aponta para:

1. o split exato CT2025 usado na Table 1 (`N=300`);
2. o comando/script usado para gerar Approach 1 / Approach 2 / Approach 3;
3. o prompt/config/modelo usado em cada approach;
4. as predições por amostra para cada approach;
5. o ficheiro de sumário que calcula METEOR;
6. checksums ou pelo menos nomes/caminhos dos ficheiros originais.

Critério mínimo para podermos manter a claim no paper:

```text
input split + script/comando + predições por amostra -> METEOR recomputado igual à Table 1 depois de arredondar
```

## Searches Úteis

No teu PC, tenta procurar por estes termos/padrões dentro do repo antigo, backups, Lambda, outputs locais, ou exports:

```bash
find . -iname '*claim*normal*' -o -iname '*meteor*' -o -iname '*comparison*'
rg -n '0\\.5583|0\\.5463|0\\.5691|N=300|n_samples.*300|METEOR|Approach 2'
rg -n 'comparison_test|claim_normalization|run_claim_normalization_ct25'
```

E em diretórios de resultados:

```text
experiments/results/claim_normalization/
claim_norm/results/
results/claim_normalization/
lambda_backup/
```

## Decisão Se Não Encontrarmos

Se não aparecer o pacote exato da Table 1, a opção segura é remover/reescrever essa claim no abstract/body. Não devemos dizer que
o método bate SOTA por `10% METEOR` sem o split `N=300` e as predições que recomputem os valores.

Fallback possível, se quisermos mencionar o diagnóstico que existe:

```text
In a broader n=1285 diagnostic comparison, the best packaged claim-normalization run reaches average METEOR 0.345; this diagnostic
is not the N=300 Table 1 result from the earlier draft.
```

Mas isso é uma claim diferente, não substitui a Table 1.
