# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oku
file_path = r"C:\Users\kadir\Downloads\Titanic\Titanic-Dataset.csv"  # Dosyanızın yolunu buraya yazın
titanic_data = pd.read_csv(file_path)

# 1. Eksik verileri kontrol et
missing_values = titanic_data.isnull().sum()

# 2. Hayatta kalma oranını hesapla
survival_rate = titanic_data['Survived'].mean()

# 3. Cinsiyetlere göre hayatta kalma oranı
survival_by_gender = titanic_data.groupby('Sex')['Survived'].mean()

# 4. Sınıflara göre hayatta kalma oranı
survival_by_class = titanic_data.groupby('Pclass')['Survived'].mean()

# 5. Yaş dağılımı
age_distribution = titanic_data['Age'].describe()

# 6. Biniş yerine göre hayatta kalma oranı
survival_by_embarked = titanic_data.groupby('Embarked')['Survived'].mean()

# 7. Görselleştirme: Cinsiyet ve Hayatta Kalma Oranı
plt.figure(figsize=(10, 6))

# Cinsiyet ve Hayatta Kalma Oranı Grafiği
#plt.subplot(2, 2, 1)
#sns.barplot(x='Sex', y='Survived', data=titanic_data)
#plt.title('Cinsiyet ve Hayatta Kalma Oranı')
#plt.ylabel('Hayatta Kalma Oranı')

# Sınıf ve Hayatta Kalma Oranı Grafiği
#plt.subplot(2, 2, 2)
#sns.barplot(x='Pclass', y='Survived', data=titanic_data)
#plt.title('Sınıf ve Hayatta Kalma Oranı')
#plt.ylabel('Hayatta Kalma Oranı')

# Yaş Dağılımı ve Hayatta Kalanların Yaş Dağılımı
'''
plt.subplot(2, 2, 3)
sns.histplot(titanic_data['Age'].dropna(), kde=True, color='blue', label='Tüm Yolcular', bins=30)
sns.histplot(titanic_data[titanic_data['Survived'] == 1]['Age'].dropna(), kde=True, color='green', label='Hayatta Kalanlar', bins=30)
plt.legend()
plt.title('Yaş Dağılımı ve Hayatta Kalanların Yaş Dağılımı')
'''
'''
# Biniş Yeri ve Hayatta Kalma Oranı Grafiği
plt.subplot(2, 2, 4)
sns.barplot(x='Embarked', y='Survived', data=titanic_data)
plt.title('Biniş Yeri ve Hayatta Kalma Oranı')
plt.ylabel('Hayatta Kalma Oranı')
'''
'''
# Grafiklerin gösterimi
plt.tight_layout()
plt.show()
'''
# Sonuçları ekrana yazdır
print("Eksik Veriler:\n", missing_values)
print("\nHayatta Kalma Oranı:", survival_rate)
print("\nCinsiyete Göre Hayatta Kalma Oranı:\n", survival_by_gender)
print("\nSınıfa Göre Hayatta Kalma Oranı:\n", survival_by_class)
print("\nYaş Dağılımı:\n", age_distribution)
print("\nBiniş Yerine Göre Hayatta Kalma Oranı:\n", survival_by_embarked)