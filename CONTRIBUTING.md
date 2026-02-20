# Contributing Guidelines

Thank you for your interest in contributing to Strands AI Functions. Whether it's a bug report, new feature, correction, or additional documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub Issues to report bugs or suggest features.

When filing an issue, please check for already tracked items to avoid duplicates.

Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used (commit ID or release version)
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment
* The Python version you're using
* The model(s) you're using (e.g., Claude Sonnet, GPT-4, etc.)


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute to. We label issues that are well-defined and ready for community contributions with the "good first issue" or "help wanted" labels.

Before starting work on any issue:
1. Check if someone is already assigned or working on it
2. Comment on the issue to express your interest and ask any clarifying questions
3. Wait for maintainer confirmation before beginning significant work


## Development Principles

Strands AI Functions follows these core principles:

1. **Formal Verification First**: Express user intent through executable pre- and post-conditions, not natural language
2. **Confined Non-Determinism**: AI-generated code is encapsulated within AI Functions with behavior controlled by verifiable conditions
3. **Verifiable Execution**: Validate execution traces against specifications before running AI-generated code
4. **Simplicity**: Keep the API simple and intuitive while maintaining powerful capabilities
5. **Type Safety**: Leverage Python's type system to catch errors early


## Development Environment

This project uses [hatchling](https://hatch.pypa.io/latest/build/#hatchling) as the build backend and [hatch](https://hatch.pypa.io/latest/) for development workflow management.

### Setting Up Your Development Environment

1. Clone the repository and enter the directory:
   ```bash
   cd ai-functions
   ```

2. Install hatch if you haven't already:
   ```bash
   pip install hatch
   ```

3. Enter the virtual environment:
   ```bash
   hatch shell
   ```

   Alternatively, install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks (if available):
   ```bash
   # If pre-commit is configured
   pre-commit install -t pre-commit -t commit-msg
   ```

### Running Tests

```bash
# Run all tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run specific test file
hatch run test tests/test_specific.py

# Run tests with hypothesis for property-based testing
pytest tests/
```

### Code Quality Tools

We use the following tools to ensure code quality:
1. **ruff** - For formatting and linting
2. **mypy** - For static type checking
3. **pytest** - For unit and integration tests
4. **hypothesis** - For property-based testing

Run these checks before submitting:

```bash
# Format code
hatch run format

# Run linter
hatch run lint

# Run type checker
hatch run typecheck

# Run all checks together
hatch run format && hatch run lint && hatch run typecheck && hatch run test
```

### Code Formatting and Style Guidelines

- **Line Length**: Maximum 120 characters
- **Docstring Style**: Google-style docstrings
- **Type Hints**: Required for all public functions and methods
- **Import Order**: Handled automatically by ruff

The tools are configured in [pyproject.toml](./pyproject.toml). Please ensure your code passes all linting and type checks before submitting a pull request.

If you're using an IDE like VS Code or PyCharm, consider configuring it to use these tools automatically:

**VS Code settings.json example:**
```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "mypy.enabled": true
}
```


## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b my-feature-branch
   ```

2. Make your changes:
   - Focus on the specific change you are contributing
   - If you also reformat all the code, it will be hard for us to focus on your change
   - Add tests for any new functionality
   - Update documentation as needed

3. Ensure your code meets quality standards:
   ```bash
   hatch run format      # Format your code
   hatch run lint        # Check linting
   hatch run typecheck   # Check types
   hatch run test        # Run tests
   ```

4. Write clear, descriptive commit messages:
   - Use the present tense ("Add feature" not "Added feature")
   - Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit the first line to 72 characters or less
   - Reference issues and pull requests liberally after the first line

5. Push to your fork and submit a pull request:
   ```bash
   git push origin my-feature-branch
   ```

6. In your pull request description:
   - Describe what the change does and why you're making it
   - Reference any related issues
   - Include any relevant context or background
   - Describe how you tested the changes

7. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

### Pull Request Checklist

Before submitting your pull request, verify:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass (`hatch run test`)
- [ ] Code is properly formatted (`hatch run format`)
- [ ] No linting errors (`hatch run lint`)
- [ ] Type checking passes (`hatch run typecheck`)
- [ ] New functionality includes tests
- [ ] Documentation has been updated if needed
- [ ] Commit messages are clear and descriptive


## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix or `_test` suffix
- Use descriptive test function names that explain what is being tested
- Include both positive and negative test cases
- Use `pytest.mark.asyncio` for async tests
- Consider using `hypothesis` for property-based tests

### Test Structure Example

```python
import pytest
from ai_functions import ai_function
from ai_functions.types import PostConditionResult


class TestAIFunction:
    """Test suite for AI function decorator."""

    @pytest.mark.asyncio
    async def test_basic_ai_function(self):
        """Test that a basic AI function executes successfully."""
        @ai_function
        async def simple_task(text: str) -> str:
            """Process the text: {text}"""

        result = await simple_task("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_post_condition_validation(self):
        """Test that post-conditions are properly validated."""
        @ai_function
        def validate_length(text: str) -> PostConditionResult:
            """Check that text is at least 10 characters."""
            return PostConditionResult(
                passed=len(text) >= 10,
                reason="Text must be at least 10 characters"
            )

        @ai_function(post_conditions=[validate_length])
        async def generate_text(topic: str) -> str:
            """Generate text about: {topic}"""

        result = await generate_text("testing")
        assert len(result) >= 10
```


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.


## Licensing

See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

By submitting a pull request, you represent that you have the right to license your contribution to AWS and the community, and agree by submitting the patch that your contributions are licensed under the Apache-2.0 license.


## Questions?

If you have questions about contributing, feel free to:
- Open a discussion in GitHub Discussions
- Ask in an existing issue
- Reach out to the maintainers

We appreciate your contributions and look forward to collaborating with you!
