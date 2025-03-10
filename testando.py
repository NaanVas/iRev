import json

# Carregar o JSON
with open("predictions_items.json", "r") as f:
    data = json.load(f)

total_usuarios = len(data)
acertos = 0

for user_id, info in data.items():
    ground_truth = set(info.get("ground_truth", []))  # Itens que o usuário realmente interagiu
    top_recommendations = set(info.get("top_recommendations", []))  # Itens recomendados

    # Verifica se algum ground_truth aparece nas recomendações
    if ground_truth & top_recommendations:  # Interseção entre os conjuntos
        acertos += 1

# Exibir resultados
print(f"Total de usuários: {total_usuarios}")
print(f"Usuários com ground_truth nas recomendações: {acertos}")
print(f"Taxa de acerto: {acertos / total_usuarios:.2%}")
