'use strict'

const { escapeHTML } = require('hexo-util')

hexo.extend.filter.register('marked:extensions', function (extensions) {
  extensions.push(
    {
      name: 'blockMath',
      level: 'block',
      start (source) {
        return source.indexOf('$$')
      },
      tokenizer (source) {
        const match = /^\s{0,3}\$\$\s*\n?([\s\S]+?)\n?\$\$(?:\n|$)/.exec(source)
        if (!match) return

        return {
          type: 'blockMath',
          raw: match[0],
          math: match[1].trim()
        }
      },
      renderer ({ math }) {
        return `<div class="math-display">\\[${escapeHTML(math)}\\]</div>\n`
      }
    },
    {
      name: 'inlineMath',
      level: 'inline',
      start (source) {
        return source.indexOf('$')
      },
      tokenizer (source) {
        const match = /^\$(?!\$)((?:\\.|[^$\n])+?)\$(?!\$)/.exec(source)
        if (!match) return

        return {
          type: 'inlineMath',
          raw: match[0],
          math: match[1].trim()
        }
      },
      renderer ({ math }) {
        return `\\(${escapeHTML(math)}\\)`
      }
    }
  )
})
