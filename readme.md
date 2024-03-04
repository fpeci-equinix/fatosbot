# Fatos Bot

[Fatos Bot] is a GenAI tool that will be able to retrieve and reason about questions taking into account internal sources such as: Confluence, Slack, Teams, Zoom meeting, documents.

## Installation

To use this code, follow these steps:

1. Clone the repository: **git clone [repository URL]**
2. Install the required dependencies: **pip install -r requirements.txt**

## Usage

1. The tool will be able to be used via Web interface where questions and files can be uploaded for query and retrieval.
2. In order for the tool to work there are set of pre-requisites, Qdrant docker needs to be run for vector database.
3. Under the model folder you need to download zephyr7b-m.gguf model.
4. Tool allows for the following types of files to be uploaded and processed: docx, pptx, pdf, json, txt.
5. In order to run the tool the following needs to be run: chainlit run app.py

Please refer to the code comments and documentation for more detailed instructions on how to use each component and customize the functionality according to your needs.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch: **git checkout -b feature/your-feature-name** .
3. Make your changes and commit them: **git commit -m 'Add some feature'** .
4. Push to the branch: **git push origin feature/your-feature-name** .
5. Submit a pull request.

## License

Apache 2.0
