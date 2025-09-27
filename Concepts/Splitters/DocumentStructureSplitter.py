from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.text_splitter import Language


doc = """
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# In a larger application, you would store data in a database
# For this example, we'll use a simple list of dictionaries
posts = [
    {'id': 1, 'title': 'First Post', 'content': 'This is the content of the first post.'},
    {'id': 2, 'title': 'Second Post', 'content': 'This is the content of the second post.'}
]

@app.route('/')
def home():
    return render_template('index.html', posts=posts)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/post/<int:post_id>')
def view_post(post_id):
    post = next((p for p in posts if p['id'] == post_id), None)
    if post:
        return render_template('post.html', post=post)
    return "Post not found", 404

@app.route('/create', methods=['GET', 'POST'])
def create_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        new_id = len(posts) + 1
        posts.append({'id': new_id, 'title': title, 'content': content})
        return redirect(url_for('home'))
    return render_template('create_post.html')

if __name__ == '__main__':
    app.run(debug=True)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0  # character-level split
)

docs = splitter.split_text(doc)
#docs = splitter.split_documents(docs). ## for document
print(docs)
