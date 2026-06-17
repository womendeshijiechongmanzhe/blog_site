'use strict'

const { slugize } = require('hexo-util')

function normalizeKnowledgeTags (input) {
  if (!input) return []

  const rawTags = Array.isArray(input)
    ? input
    : Object.keys(input).map(key => input[key] || key)

  const seen = new Set()
  const tags = []

  rawTags.forEach(item => {
    const name = typeof item === 'string' ? item : item && item.name
    const normalized = String(name || '').trim()

    if (!normalized || seen.has(normalized)) return

    seen.add(normalized)
    tags.push(normalized)
  })

  return tags
}

function getKnowledgeTags (hexo) {
  return normalizeKnowledgeTags(hexo.config.knowledge_tags)
}

function getTagPath (hexo, tagName) {
  let tagDir = hexo.config.tag_dir || 'tags'
  if (!tagDir.endsWith('/')) tagDir += '/'

  const mappedName = hexo.config.tag_map && Object.prototype.hasOwnProperty.call(hexo.config.tag_map, tagName)
    ? hexo.config.tag_map[tagName] || tagName
    : tagName

  return `${tagDir}${slugize(mappedName, { transform: hexo.config.filename_case })}/`
}

hexo.extend.filter.register('before_generate', function () {
  const Tag = this.model('Tag')

  getKnowledgeTags(this).forEach(name => {
    if (!Tag.findOne({ name })) {
      Tag.insert({ name })
    }
  })

  this.locals.set('tags', () => {
    const manualTags = new Set(getKnowledgeTags(this))

    return this.model('Tag').filter(tag => tag.length || manualTags.has(tag.name))
  })
})

hexo.extend.generator.register('knowledge-empty-tags', function (locals) {
  const pages = []

  getKnowledgeTags(this).forEach(name => {
    const tag = locals.tags.findOne({ name })

    if (tag && tag.length) return

    const base = tag ? tag.path : getTagPath(this, name)

    pages.push({
      path: base,
      layout: ['tag', 'archive', 'index'],
      data: {
        base,
        total: 1,
        current: 1,
        current_url: base,
        posts: [],
        prev: 0,
        prev_link: '',
        next: 0,
        next_link: '',
        tag: name
      }
    })
  })

  return pages
})
