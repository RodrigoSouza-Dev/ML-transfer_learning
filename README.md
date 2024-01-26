# ML-transfer_learning
Treinamento de redes neurais com transfer learning.

Este código carrega o modelo pré-treinado BERT e ajusta-o para a tarefa específica de classificação de sentimentos. Ele usa uma pequena quantidade de dados de exemplo para treinamento e teste. A avaliação final é realizada usando um relatório de classificação que mostra métricas como precisão, recall e F1-score.


# Aprendizado por tansferencia:

O aprendizado por transferência (TL) é uma técnica de aprendizado de máquina (ML) que **reutiliza um modelo treinado projetado para uma tarefa específica para realizar uma tarefa diferente, mas relacionada**. O conhecimento adquirido da tarefa um é assim transferido para o segundo modelo que se concentra na nova tarefa.

O TL pode ser usado para melhorar o desempenho de um modelo ML em uma nova tarefa, mesmo quando há pouca ou nenhuma informação disponível sobre essa tarefa. Isso ocorre porque o modelo pré-treinado já aprendeu a identificar padrões e características relevantes para as tarefas relacionadas.

As aplicações do TL em ML são amplas e incluem:

Visão computacional: O TL é usado para tarefas como detecção de objetos, reconhecimento facial e classificação de imagens. Por exemplo, um modelo pré-treinado para detectar rostos humanos pode ser usado para treinar um novo modelo para detectar rostos de cães.
Processamento de linguagem natural: O TL é usado para tarefas como tradução de idiomas, compreensão de linguagem natural e geração de texto. Por exemplo, um modelo pré-treinado para gerar texto pode ser usado para treinar um novo modelo para gerar resumos de documentos.
Robótica: O TL é usado para tarefas como navegação autônoma, manipulação de objetos e reconhecimento de objetos. Por exemplo, um modelo pré-treinado para detectar objetos pode ser usado para treinar um novo modelo para detectar objetos em um ambiente específico.
Os principais métodos de TL incluem:

Envolvimento: O modelo pré-treinado é congelado e as camadas finais são refeitas para a nova tarefa.
Ajuste fino: O modelo pré-treinado é ajustado para a nova tarefa, permitindo que algumas ou todas as camadas sejam modificadas.
Embarquamento: As camadas intermediárias do modelo pré-treinado são usadas como recursos para treinar um novo modelo.
O TL é uma técnica poderosa que pode ser usada para melhorar o desempenho de modelos ML em uma variedade de tarefas.
