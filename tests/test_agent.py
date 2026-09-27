"""Offline checks: no real API keys or provider requests."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from streamlit.testing.v1 import AppTest
import agent_core as core

class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

class AgentTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, **kwargs):
        return core.Settings(keys={'GROQ_API_KEY': 'test-groq', 'GEMINI_API_KEY': 'test-gemini'}, **kwargs)

    async def test_real_graph_and_history(self):
        model = ToolModel(responses=[AIMessage(content='Final answer')])
        with patch.object(core, 'build_model', return_value=model):
            result = await core.ask_agent_async(
                [{'role': 'user', 'content': 'Earlier question'},
                 {'role': 'assistant', 'content': 'Earlier answer', 'provider': 'Gemini'}],
                'A follow-up', 'Auto', self.settings())
        self.assertEqual(result['answer'], 'Final answer')
        self.assertEqual(result['provider'], 'Groq')

    async def test_graph_calls_search_tool(self):
        model = ToolModel(responses=[
            AIMessage(content='', tool_calls=[{'name': 'search_web', 'args': {'query': 'test'}, 'id': 'call1'}]),
            AIMessage(content='Found a source: https://example.test')])
        calls = []
        def respond(request):
            calls.append(request.url.host)
            return httpx.Response(200, json={'results': [{'title': 'Source', 'url': 'https://example.test', 'content': 'Fact'}]})
        real_client = httpx.AsyncClient
        settings = self.settings()
        settings.keys['TAVILY_API_KEY'] = 'mock-tool-key'
        with patch.object(core, 'build_model', return_value=model), patch.object(
            core.httpx, 'AsyncClient', side_effect=lambda **kw: real_client(transport=httpx.MockTransport(respond))):
            result = await core.ask_agent_async([], 'Search for this', 'Groq', settings)
        self.assertIn('example.test', result['answer'])
        self.assertEqual(calls, ['api.tavily.com'])

    async def test_quota_fallback_preserves_history(self):
        calls = []
        async def run(provider, settings, messages):
            calls.append((provider, messages))
            if provider == 'Groq':
                raise RuntimeError('quota exceeded; secret-should-not-appear')
            return 'OK'
        with patch.object(core, 'run_provider', side_effect=run):
            result = await core.ask_agent_async([], 'Question', 'Auto', self.settings())
        self.assertEqual([p for p, _ in calls], ['Groq', 'Gemini'])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(result['provider'], 'Gemini')
        self.assertNotIn('secret', str(result))

    async def test_timeout_cancels_before_fallback(self):
        cancelled = []
        async def run(provider, settings, messages):
            if provider == 'Groq':
                try:
                    await asyncio.sleep(5)
                finally:
                    cancelled.append(provider)
            return 'OK'
        with patch.object(core, 'run_provider', side_effect=run):
            result = await core.ask_agent_async([], 'Question', 'Auto', self.settings(cloud_timeout=.02))
        self.assertEqual(cancelled, ['Groq'])
        self.assertEqual(result['provider'], 'Gemini')

    async def test_manual_provider_never_falls_back(self):
        with patch.object(core, 'run_provider', side_effect=RuntimeError('quota')) as run:
            with self.assertRaises(core.AgentFailure):
                await core.ask_agent_async([], 'Question', 'Groq', self.settings())
        self.assertEqual(run.call_count, 1)

    async def test_all_fail_and_no_secret_in_error(self):
        with patch.object(core, 'run_provider', side_effect=RuntimeError('test-secret-payload')):
            with self.assertRaises(core.AgentFailure) as error:
                await core.ask_agent_async([], 'Question', 'Auto', self.settings())
        self.assertEqual(len(error.exception.attempts), 2)
        self.assertNotIn('test-secret-payload', str(error.exception))

    async def test_unconfigured_and_disabled_skipped(self):
        self.assertEqual(core.candidates('Auto', core.Settings()), [])
        self.assertEqual(core.candidates('Auto', self.settings(auto_order=('OpenAI', 'Gemini'))), ['Gemini'])
        self.assertEqual(core.candidates('Ollama', core.Settings()), ['Ollama'])
        with self.assertRaises(ValueError):
            core.candidates('OpenAI', self.settings())

    async def test_refusal_returned_without_fallback(self):
        with patch.object(core, 'run_provider', return_value='I cannot help with that.') as run:
            result = await core.ask_agent_async([], 'Question', 'Auto', self.settings())
        self.assertEqual(run.call_count, 1)
        self.assertIn('cannot help', result['answer'])

    async def test_text_excludes_thought_blocks(self):
        msg = AIMessage(content=[{'type': 'reasoning', 'text': 'private'},
            {'type': 'text', 'text': 'hidden', 'thought': True}, {'type': 'text', 'text': 'Visible'}])
        self.assertEqual(core.text_content(msg), 'Visible')

class UiTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'apps.py'), default_timeout=15)

    def test_new_provider_key_chat_and_clear(self):
        result = {'answer': 'Mock answer', 'provider': 'Gemini', 'model': 'gemini-2.5-flash',
                  'seconds': .2, 'attempts': [{'provider': 'Groq', 'detail': 'Quota reached'}]}
        with patch.object(core, 'load_defaults', return_value={}), patch.object(core, 'ask_agent', return_value=result):
            at = self.app().run()
            self.assertEqual(len(at.exception), 0)
            self.assertIn('Gemini', at.selectbox[0].options)
            at.text_input(key='key_GEMINI_API_KEY').set_value('test-key').run()
            at.chat_input[0].set_value('Hello').run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state['messages'][-1]['content'], 'Mock answer')
            self.assertNotIn('test-key', str(at.session_state['messages']))
            next(b for b in at.button if b.label == 'Clear conversation').click().run()
            self.assertEqual(at.session_state['messages'], [])

    def test_failed_question_retry(self):
        result = {'answer': 'Recovered', 'provider': 'Groq', 'model': 'mock', 'seconds': .1, 'attempts': []}
        with patch.object(core, 'load_defaults', return_value={}), patch.object(core, 'ask_agent',
             side_effect=[core.AgentFailure([{'provider': 'Groq', 'detail': 'Timeout'}]), result]):
            at = self.app().run()
            at.chat_input[0].set_value('Retry me').run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state['messages'], [])
            next(b for b in at.button if b.label == 'Retry with current settings').click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state['messages']), 2)
            self.assertEqual(at.session_state['messages'][-1]['content'], 'Recovered')
            self.assertEqual(at.session_state['failed_question'], '')

if __name__ == '__main__':
    unittest.main()
