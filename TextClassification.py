import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt


# Test.csv dosyasını okuyarak bir pandas DataFrame oluşturuyoruz. Ardından,
# metinleri ve etiketleri ilgili değişkenlere atıyoruz.

data = pd.read_csv('Test.csv')

# data = data.head(100)
# data = data.iloc[100:200]
data = data.iloc[200:300]
# data = data.iloc[300:400]
# data = data.iloc[400:500]
texts = data['text']
labels = data['label']


# Veri kümesini eğitim ve test setlerine bölüyoruz. (test seti %20 boyutunda).
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)


# fitness fonksiyonunu tanımlarız. Bu fonksiyon, bir bireyin sınıflandırma başarısını hesaplar.
def fitness(individual, texts, labels):
    # Doğru tahminlerin sayısı için correct sayacını başlatıyoruz.
    correct = 0

    # Metin ve etiket çiftleri üzerinde döngü başlatıyoruz.
    for text, label in zip(texts, labels):
        words = text.split()
        # Metindeki pozitif ve negatif kelimelerin sayısını hesaplarız.
        pos_count = sum([1 for word in words if word in individual[:N//2]])
        neg_count = sum([1 for word in words if word in individual[N//2:]])
        # Tahmini etiketi belirleriz. Eğer pozitif kelime sayısı negatif kelime sayısından fazlaysa 1, aksi takdirde 0 olarak atanır.
        # Eşitse, rastgele 0 veya 1 atanır.
        prediction = 1 if pos_count > neg_count else 0 if neg_count > pos_count else random.randint(
            0, 1)
        if prediction == label:  # Tahmin doğruysa, correct değişkenini artırırız.
            correct += 1
    # Doğru tahminlerin oranını döndürürüz (sınıflandırma başarısı).
    return correct / len(labels)


# İki ebeveyn arasında tek noktalı çaprazlama gerçekleştiren crossover fonksiyonu.
def crossover(parent1, parent2):
    crossover_point = random.randint(1, N - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2  # Çaprazlama işlemi ile iki çocuk birey oluşturulur.


# Bireylerin genlerinde küçük rastgele değişiklikler yaparak popülasyonun çeşitliliğini artırır.
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_index] = individual[swap_index], individual[i]
    return individual


# Hiperparametreler
N = 10  # Her bireyin kelime listesi uzunluğu
population_size = 50  # Toplam birey sayısı
mutation_rate = 0.1  # Mutasyon oranı
generations = 100

# Popülasyonu alır.
all_words = set()
for text in X_train:
    words = text.split()
    all_words.update(words)
all_words = list(all_words)

# Genetik algoritmanın uygulanacağı şekilde başlangıç popülasyonu hazırlar.
population = [random.sample(all_words, N) for _ in range(population_size)]


best_individual = None
best_fitness = 0
avg_fitness = []

# Hazırlanan popülasyona genetik algoritmanın uygulanması
for gen in range(generations):
    # Fitness hesabı
    fitness_values = [fitness(ind, X_train, y_train) for ind in population]

    # En iyi bireyin ve ortalama fitness değerinin hesaplanması
    max_index = np.argmax(fitness_values)
    if fitness_values[max_index] > best_fitness:
        best_fitness = fitness_values[max_index]
        best_individual = population[max_index]

    avg_fitness.append(np.mean(fitness_values))

    # Selection, iyi olanların seçilip bir sonraki jenerasyonda ebeveyn olması sağlanır.
    parents = random.choices(
        population, weights=fitness_values, k=population_size // 2 * 2)

    # Crossover, ebeveynler çarprazların ve iki çocuk üretilir.
    children = []
    for p1, p2 in zip(parents[::2], parents[1::2]):
        c1, c2 = crossover(p1, p2)
        children.extend([c1, c2])

    # Mutation, oluşturulan çocuklara c1, c2 mutasyon fonksiyonu uygulanır.
    for child in children:
        mutate(child, mutation_rate)

    population = children


print("N:", N)
print("population_size:", population_size)
print("mutation_rate:", mutation_rate)
print("generations:", generations)

# En iyi bireyin fitness değeri (Test Accuracy)
test_accuracy = fitness(best_individual, X_test, y_test)
print(f"Best individual's test accuracy: {test_accuracy:.2f}")

# Bireylerin sınıflandırma başarısı ve ortalama başarısı
print(f"Best individual's classification success: {best_fitness:.2f}")
print(
    f"Final generation's average success of individuals: {avg_fitness[-1]:.2f}")

# Jenerasyondaki ortalama fitness değerinin değişim grafiği
plt.plot(avg_fitness)
plt.title(
    f"Average Fitness Over Generations (N={N}, Population Size={population_size}, Mutation Rate={mutation_rate})")
# X -> Jenerasyonlar Y -> Avg Fitness Değeri
plt.xlabel(f"Generations (Total: {generations})")
plt.ylabel("Average Fitness")

# Raporda istenen değerler de grafikte çıktı olarak verilir.
text_x = 0.05
text_y = 0.9
plt.gca().annotate(f"Best individual's test accuracy: {test_accuracy:.2f}", xy=(
    text_x, text_y), xycoords='axes fraction')
text_y -= 0.05
plt.gca().annotate(f"Best individual's classification success: {best_fitness:.2f}", xy=(
    text_x, text_y), xycoords='axes fraction')
text_y -= 0.05
plt.gca().annotate(f"Final generation's average success of individuals: {avg_fitness[-1]:.2f}", xy=(
    text_x, text_y), xycoords='axes fraction')

# En iyi birey:
plt.text(0, -0.1,
         f"Best word list: {best_individual}", fontsize=8, transform=plt.gca().transAxes)


plt.show()
