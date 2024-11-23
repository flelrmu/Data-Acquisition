import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Market Analysis", page_icon="💻", layout="wide")

page = st.sidebar.selectbox("", ["Home", "Analisis Data"])

if page == "Home":
    st.title("Bagaimana AI Mengubah Pasar Kerja?🤖")
    st.image(
        "https://bizaccenknnect.com/wp-content/uploads/2023/09/psd-1.png",
        caption="Pasar Kerja Berbasis AI",
        use_container_width=True
    )

    st.header("📖 Latar Belakang")
    st.write(
        """
        Pasar kerja global sedang mengalami transformasi signifikan yang dipicu oleh kemajuan kecerdasan buatan (AI). 
        Teknologi ini menciptakan tantangan dan peluang baru di berbagai sektor. Otomatisasi yang didorong AI 
        menggantikan pekerjaan rutin, tetapi juga membuka peluang untuk pekerjaan dengan keterampilan khusus 
        seperti analisis data dan pengembangan algoritma.
        """
    )

    st.header("🎯 Tujuan")
    st.write(
        """
        1. **Mengelompokkan pekerjaan** berdasarkan tingkat gaji (Salary_USD) dan faktor lain seperti industri dan risiko otomatisasi.
        2. **Mengidentifikasi pola tersembunyi** untuk mendukung keputusan strategis.
        3. **Menyediakan wawasan baru** melalui visualisasi data untuk memahami karakteristik pasar kerja berbasis AI.
        4. **Membagi pasar kerja berbasis AI menjadi beberapa segmen** untuk memahami karakteristik unik masing-masing segmen.
        5. **Membantu menemukan wawasan baru** di pasar kerja berbasis AI melalui visualisasi data.
        """
    )

elif page == "Analisis Data":
    st.title("Analisis Pasar Kerja Berbasis AI 📊")
    st.subheader("Upload Dataset")

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file XLSX", type="xlsx")
    if uploaded_file:
        st.markdown("---")
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.dataframe(df)

        # Info Dataset
        if st.button("Info Dataset"):
            with st.expander("Informasi Dataset", expanded=True):
                st.write("### Info Umum Dataset")
                st.write(f"Jumlah Baris: {df.shape[0]}")
                st.write(f"Jumlah Kolom: {df.shape[1]}")
                st.write("### Tipe Data")
                st.write(df.dtypes)

        # Preprocessing Data
        st.subheader("Preprocessing Data")
        if st.button("Lakukan Preprocessing"):
            # Menampilkan jumlah nilai NaN sebelum menghapusnya
            st.write("### Nilai yang Hilang (NaN) Sebelum Preprocessing:")
            st.write(df.isna().sum())  # Menampilkan jumlah nilai NaN untuk setiap kolom

            # Menghapus nilai NaN
            df = df.dropna()

            # Menampilkan jumlah nilai NaN setelah penghapusan
            st.write("### Nilai yang Hilang (NaN) Setelah Preprocessing:")
            st.write(df.isna().sum())  # Memeriksa kembali apakah masih ada NaN

            # Normalisasi kolom Salary_USD
            scaler = StandardScaler()
            columns_to_scale = ["Salary_USD"]
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

            # Menyimpan data yang telah diproses dalam session state
            st.session_state["preprocessed_data"] = df
            
            # Menampilkan data yang sudah diproses
            st.write("### Data setelah preprocessing (normalisasi gaji):")
            st.dataframe(df)

        # Clustering
        st.subheader("Analisis Data - Clustering")
        if st.session_state.get("preprocessed_data") is not None:
            k = st.slider("Pilih Jumlah Cluster", 2, 10, 3)
            if st.button("Lakukan Clustering (K-Means)"):
                # Gunakan data yang sudah diproses
                df = st.session_state["preprocessed_data"].copy()
                model = KMeans(n_clusters=k)
                df["Cluster"] = model.fit_predict(df[["Salary_USD"]])
                st.session_state["clustered_data"] = df  # Simpan hasil clustering

                st.write(f"Data dengan Cluster K-Means (k={k}):")
                st.dataframe(df)
                st.write("Centroid Cluster:")
                st.write(pd.DataFrame(model.cluster_centers_, columns=["Salary_USD"]))

                # Tampilkan tabel rincian untuk setiap cluster
                for cluster_num in range(k):
                    st.subheader(f"Rincian Cluster {cluster_num+1}")
                    cluster_data = df[df["Cluster"] == cluster_num]
                    st.dataframe(cluster_data)

                # Menambahkan penjelasan untuk setiap cluster
                st.write("### Penjelasan Cluster:")
                cluster_explanations = {
                    0: "Pekerjaan dengan risiko otomatisasi rendah, gaji tinggi, dan keterampilan teknis tinggi—berpotensi untuk berkembang di masa depan.",
                    1: "Pekerjaan dengan risiko otomatisasi sedang, gaji menengah, dan keterampilan teknis sedang—cocok untuk profesi yang membutuhkan adaptasi teknologi.",
                    2: "Pekerjaan dengan risiko otomatisasi tinggi, gaji rendah, dan keterampilan teknis rendah—berisiko berkurang seiring perkembangan teknologi otomatisasi.",
                    3: " Pekerjaan dengan risiko otomatisasi rendah, gaji menengah, dan keterampilan manajerial—berpotensi stabil meskipun teknologi berkembang.",
                    4: "Pekerjaan dengan risiko otomatisasi sangat rendah, gaji tinggi, dan keterampilan kepemimpinan—biasanya di level eksekutif atau strategis.",
                    5: "Pekerjaan dengan risiko otomatisasi sedang, gaji menengah, dan keterampilan manajerial—membutuhkan pemahaman terhadap proses bisnis dan manajemen.",
                    6: "Pekerjaan dengan risiko otomatisasi rendah, gaji sedang, dan keterampilan komunikasi—berkaitan dengan posisi pelayanan atau dukungan pelanggan.",
                    7: "Pekerjaan dengan risiko otomatisasi tinggi, gaji rendah, dan keterampilan manual—terkait dengan pekerjaan fisik atau rutin.",
                    8: "Pekerjaan dengan risiko otomatisasi sangat rendah, gaji sangat tinggi, dan keterampilan teknis tingkat tinggi—biasanya dalam sektor penelitian atau inovasi.",
                    9: "Pekerjaan dengan risiko otomatisasi tinggi, gaji menengah, dan keterampilan teknis dalam analisis data—terkait dengan pekerjaan yang memanfaatkan teknologi tetapi berisiko tergantikan dengan AI."
                }

                for i in range(k):
                    st.write(f"Cluster {i+1}: {cluster_explanations.get(i, 'Penjelasan tidak tersedia')}")


        # Visualisasi Data
if "clustered_data" in st.session_state:
    st.subheader("Visualisasi Data")
    df = st.session_state["clustered_data"]
    visual_options = [
        "Distribusi Gaji berdasarkan Jabatan Pekerjaan",
        "Perbandingan Gaji antar Industri",
        "Hubungan antara Risiko Otomatisasi dan Gaji",
        "Hubungan antara Ukuran Perusahaan dan Risiko Otomatisasi",
        "Distribusi Lokasi Kerja Berdasarkan Remote Friendly dan Industri",
        "Persentase Pekerjaan yang Mendukung Kerja Jarak Jauh",
        "Distribusi Gaji berdasarkan Ukuran Perusahaan",
        "Hubungan Proyeksi Pertumbuhan dengan Tingkat Adopsi AI",
        "Keterampilan yang Paling Banyak Dibutuhkan",
        "Distribusi Gaji Berdasarkan Lokasi",
        "Rata-rata Gaji untuk Setiap Tingkat Adopsi AI",
        "Perbandingan Risiko Otomatisasi di Berbagai Lokasi",
        "Proyeksi Pertumbuhan Pekerjaan Berdasarkan Industri",
        "Frekuensi Lokasi Kerja berdasarkan Remote Friendly",
        "Distribusi Pekerjaan Berdasarkan Ukuran Perusahaan",
        "Hubungan Gaji dengan Jumlah Keterampilan yang Dibutuhkan",
        "Distribusi Jabatan Berdasarkan Industri",
        "Korelasi antara Gaji dan Risiko Otomatisasi",
        "Frekuensi Jabatan yang Mendukung Kerja Remote",
        "Distribusi Gaji berdasarkan Proyeksi Pertumbuhan Pekerjaan",
        "Frekuensi Jabatan Berdasarkan Industri",
    ]
    viz_choice = st.selectbox("Pilih Visualisasi", visual_options)

    if viz_choice == visual_options[0]:
        fig = px.histogram(
            df, 
            x='Salary_USD', 
            color='Job_Title', 
            title="Distribusi Gaji berdasarkan Jabatan Pekerjaan",
            nbins=30, 
            barmode='overlay'
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik ini menunjukkan distribusi gaji berdasarkan jabatan pekerjaan. 
            Setiap warna mewakili jabatan yang berbeda. 
            Dapat dilihat bahwa jabatan tertentu memiliki distribusi gaji yang lebih tinggi, 
            menandakan pekerjaan tersebut cenderung memiliki gaji yang lebih besar di pasar kerja.
        """)

    elif viz_choice == visual_options[1]:
        fig = px.bar(df, x='Industry', y='Salary_USD', color='Cluster', title="Perbandingan Gaji antar Industri")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini membandingkan gaji rata-rata berdasarkan industri, dengan warna yang menunjukkan cluster yang berbeda. 
            Ini memberikan gambaran tentang bagaimana industri-industri berbeda dalam hal tingkat gaji pekerjaannya.
        """)

    elif viz_choice == visual_options[2]:
        fig = px.scatter(df, x='Automation_Risk', y='Salary_USD', color='Cluster', title="Hubungan Risiko Otomatisasi dan Gaji")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik pencar ini menunjukkan hubungan antara risiko otomatisasi dan gaji. 
            Titik-titik yang lebih tinggi di sumbu Y menunjukkan gaji yang lebih besar, 
            sedangkan posisi di sumbu X menunjukkan tingkat risiko otomatisasi. 
            Warna yang berbeda mewakili cluster yang teridentifikasi dalam data.
        """)

    elif viz_choice == visual_options[3]:
        df['Salary_Size'] = df['Salary_USD'] + abs(df['Salary_USD'].min()) + 1
        fig = px.scatter(
            df, x='Company_Size', y='Automation_Risk', size='Salary_Size', color='Industry',
            title="Hubungan antara Ukuran Perusahaan dan Risiko Otomatisasi",
            hover_data=['Job_Title']
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik pencar ini menggambarkan hubungan antara ukuran perusahaan dan risiko otomatisasi. 
            Ukuran titik menunjukkan besar gaji yang diterima di perusahaan tersebut, 
            sementara warna mewakili industri tempat perusahaan beroperasi.
        """)

    elif viz_choice == visual_options[4]:
        fig = px.bar(
            df, x='Location', color='Industry', barmode='stack',
            facet_col='Remote_Friendly', title="Distribusi Lokasi Kerja Berdasarkan Remote Friendly dan Industri"
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik ini memperlihatkan distribusi lokasi pekerjaan yang berbeda berdasarkan dua kategori: 
            industri dan apakah pekerjaan tersebut mendukung kerja jarak jauh. 
            Setiap batang menunjukkan jumlah pekerjaan di lokasi tertentu yang dibagi berdasarkan industri.
        """)

    elif viz_choice == visual_options[5]:
        fig = px.pie(df, names='Remote_Friendly', title="Persentase Pekerjaan yang Mendukung Kerja Jarak Jauh")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik pie ini menunjukkan persentase pekerjaan yang mendukung kerja jarak jauh dibandingkan dengan pekerjaan yang tidak. 
            Ini memberikan gambaran yang jelas tentang sejauh mana industri menerima kerja jarak jauh.
        """)

    elif viz_choice == visual_options[6]:
        fig = px.box(df, x='Company_Size', y='Salary_USD', title="Distribusi Gaji berdasarkan Ukuran Perusahaan")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik boxplot ini menggambarkan distribusi gaji berdasarkan ukuran perusahaan. 
            Dari boxplot, kita dapat melihat variasi gaji untuk perusahaan dengan ukuran yang berbeda. 
            Titik luar (outliers) menunjukkan gaji yang jauh lebih tinggi atau lebih rendah dari mayoritas.
        """)

    elif viz_choice == visual_options[7]:
        fig = px.bar(df, x='AI_Adoption_Level', y='Job_Growth_Projection', color='Industry', title="Hubungan Proyeksi Pertumbuhan dengan Tingkat Adopsi AI")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini menunjukkan hubungan antara tingkat adopsi AI dan proyeksi pertumbuhan pekerjaan, 
            dengan warna yang menunjukkan industri. 
            Ini memberikan wawasan tentang bagaimana industri-industri tertentu melihat perkembangan pekerjaan 
            terkait dengan adopsi AI.
        """)

    elif viz_choice == visual_options[8]:
        skill_words = " ".join(df['Required_Skills'].dropna())
        wordcloud = WordCloud(width=800, height=400).generate(skill_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.write("""
            **Penjelasan:**
            Word cloud ini menampilkan keterampilan yang paling banyak dibutuhkan dalam pekerjaan yang ada. 
            Kata-kata yang lebih besar menunjukkan keterampilan yang lebih sering disebutkan dalam data, 
            memberikan gambaran tentang keterampilan penting yang dicari oleh pemberi kerja.
        """)

    elif viz_choice == visual_options[9]:
        fig = px.histogram(
            df, 
            x='Location', 
            y='Salary_USD', 
            title="Distribusi Gaji Berdasarkan Lokasi",
            nbins=30, 
            histfunc='avg' 
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik histogram ini menunjukkan distribusi rata-rata gaji berdasarkan lokasi pekerjaan. 
            Dengan menggunakan histogram, kita dapat melihat tren gaji di berbagai lokasi dan sektor.
        """)

    elif viz_choice == visual_options[10]:
        avg_salary = df.groupby('AI_Adoption_Level')['Salary_USD'].mean().reset_index()
        fig = px.bar(avg_salary, x='AI_Adoption_Level', y='Salary_USD', title="Rata-rata Gaji untuk Setiap Tingkat Adopsi AI")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini menunjukkan rata-rata gaji berdasarkan tingkat adopsi AI di berbagai industri. 
            Dengan melihat grafik ini, kita bisa memahami bagaimana industri yang lebih banyak mengadopsi AI membayar lebih tinggi.
        """)

    elif viz_choice == visual_options[11]:
        fig = px.bar(df, x='Location', y='Automation_Risk', color='Industry', barmode='group',
                     title="Perbandingan Risiko Otomatisasi di Berbagai Lokasi")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini membandingkan tingkat risiko otomatisasi di berbagai lokasi, dengan warna yang menunjukkan industri yang terlibat. 
            Dapat dilihat bagaimana lokasi-lokasi tertentu menghadapi risiko otomatisasi yang lebih tinggi.
        """)

    elif viz_choice == visual_options[12]:
        fig = px.bar(df, x='Industry', y='Job_Growth_Projection', color='Industry',
                     title="Proyeksi Pertumbuhan Pekerjaan Berdasarkan Industri")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini menunjukkan proyeksi pertumbuhan pekerjaan berdasarkan industri. 
            Ini memberikan wawasan tentang industri mana yang diperkirakan akan mengalami peningkatan jumlah pekerjaan.
        """)

    elif viz_choice == visual_options[13]:
        remote_counts = df['Remote_Friendly'].value_counts().reset_index()
        remote_counts.columns = ['Remote_Friendly', 'Count']

        fig = px.bar(
            remote_counts, 
            x='Remote_Friendly', 
            y='Count', 
            color='Remote_Friendly',
            title="Frekuensi Lokasi Kerja berdasarkan Remote Friendly",
            text='Count', 
            color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"] 
        )

        fig.update_layout(
            xaxis_title="Tipe Lokasi Kerja", 
            yaxis_title="Jumlah Frekuensi", 
            font=dict(size=14),  
            title_font=dict(size=18),  
            legend_title="Remote Friendly",  
        )

        fig.update_traces(
            texttemplate='%{text}',  
            textposition='outside',  
            marker=dict(line=dict(width=2, color='DarkSlateGrey'))  
        )

        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini memperlihatkan frekuensi lokasi kerja berdasarkan apakah pekerjaan tersebut mendukung kerja jarak jauh atau tidak. 
            Dapat dilihat seberapa banyak pekerjaan yang tersedia di kategori ini di seluruh lokasi.
        """)

    elif viz_choice == visual_options[14]:
        fig = px.pie(df, names='Company_Size', title="Distribusi Pekerjaan Berdasarkan Ukuran Perusahaan")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik pie ini menunjukkan distribusi pekerjaan berdasarkan ukuran perusahaan. 
            Dengan ini, kita bisa melihat persentase pekerjaan yang berasal dari perusahaan kecil, menengah, atau besar.
        """)

    elif viz_choice == visual_options[15]:
        df['Skill_Count'] = df['Required_Skills'].str.split().apply(len)
        fig = px.scatter(
            df, x='Skill_Count', y='Salary_USD', color='Industry',
            title="Hubungan Gaji dengan Jumlah Keterampilan yang Dibutuhkan",
            hover_data=['Job_Title']
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik pencar ini menunjukkan hubungan antara jumlah keterampilan yang dibutuhkan dalam pekerjaan dan gaji yang diterima. 
            Titik-titik yang lebih tinggi di sumbu Y menunjukkan pekerjaan dengan gaji lebih tinggi, sementara sumbu X menunjukkan jumlah keterampilan yang dibutuhkan.
        """)

    elif viz_choice == visual_options[16]:
        fig = px.bar(df, x='Industry', y='Job_Title', color='Industry', text='Job_Title', barmode='stack',
                     title="Distribusi Jabatan Berdasarkan Industri")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini menunjukkan distribusi jabatan pekerjaan di berbagai industri. 
            Masing-masing batang menunjukkan jumlah jabatan dalam kategori industri tertentu.
        """)

    elif viz_choice == visual_options[17]:
        df['Automation_Risk_Numeric'] = df['Automation_Risk'].map({'Low': 1, 'Medium': 2, 'High': 3})

        correlation_data = df[['Salary_USD', 'Automation_Risk_Numeric']].corr()

        fig = px.imshow(correlation_data, text_auto=True, title="Korelasi antara Gaji dan Risiko Otomatisasi")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik ini menunjukkan korelasi antara gaji dan tingkat risiko otomatisasi. 
            Nilai korelasi yang lebih tinggi menunjukkan hubungan yang lebih kuat antara dua variabel ini.
        """)

    elif viz_choice == visual_options[18]:
        remote_jobs = df[df['Remote_Friendly'] == 'Yes']['Job_Title'].value_counts().reset_index()
        remote_jobs.columns = ['Job_Title', 'Count']
        fig = px.bar(remote_jobs, x='Job_Title', y='Count', title="Frekuensi Jabatan yang Mendukung Kerja Remote")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik batang ini menunjukkan frekuensi jabatan yang mendukung kerja jarak jauh. 
            Dapat dilihat jabatan mana yang lebih sering ditemukan di pekerjaan remote.
        """)

    elif viz_choice == visual_options[19]:
        fig = px.violin(df, x='Job_Growth_Projection', y='Salary_USD', color='Cluster',
                        title="Distribusi Gaji berdasarkan Proyeksi Pertumbuhan Pekerjaan")
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik violin ini menunjukkan distribusi gaji berdasarkan proyeksi pertumbuhan pekerjaan. 
            Tiap band menunjukkan distribusi gaji di kelompok proyeksi pertumbuhan pekerjaan yang berbeda, yang membantu memahami variasi gaji.
        """)

    elif viz_choice == visual_options[20]:
        pivot_data = df.pivot_table(
            index='Industry',
            columns='Job_Title',
            values='Salary_USD',
            aggfunc='count',  
            fill_value=0 
        )

        fig = px.imshow(
            pivot_data,
            color_continuous_scale='Viridis', 
            labels=dict(x="Job Title", y="Industry", color="Frequency"),
            title="Frekuensi Jabatan Berdasarkan Industri"
        )
        st.plotly_chart(fig)
        st.write("""
            **Penjelasan:**
            Grafik ini menunjukkan frekuensi jabatan berdasarkan industri, dengan warna yang mencerminkan jumlah jabatan di industri tertentu. 
            Ini memberikan gambaran tentang industri mana yang memiliki lebih banyak jenis pekerjaan yang tersedia.
        """)
