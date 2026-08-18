#!/usr/bin/env kotlin
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-serialization-json-jvm:1.8.1")

import java.io.File
import java.util.Locale
import kotlin.system.exitProcess
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

val USAGE = """
Usage: kotlin slop-detector.main.kts <file|glob|directory>... [--images] [--verbose]

  Accepts .webarchive and Markdown (.md, .markdown, .mdown, .mkd) files.
  Several inputs, a glob such as '*.md', or a directory are scored together
  as one collective answer; --verbose breaks it down per file.
""".trim()
val KNOWN_FLAGS = setOf("--images", "--verbose")

val repoRoot: File = File(System.getProperty("user.dir")).absoluteFile
val skipBootstrap: Boolean = System.getenv("SLOP_DETECTOR_SKIP_BOOTSTRAP") == "1"
val venvPython: File = File(repoRoot, ".venv/bin/python3")
val modelCache: File = File(repoRoot, ".model-cache")

fun fail(message: String, code: Int): Nothing {
    System.err.println(message)
    exitProcess(code)
}

fun run(command: List<String>, directory: File, description: String) {
    val exit = ProcessBuilder(command)
        .directory(directory)
        .redirectOutput(ProcessBuilder.Redirect.INHERIT)
        .redirectError(ProcessBuilder.Redirect.INHERIT)
        .start()
        .waitFor()
    if (exit != 0) fail("$description failed with exit code $exit", 1)
}

/** Create the project-local virtual environment and install worker requirements. */
fun bootstrapPython() {
    val requirements = File(repoRoot, "worker/requirements.txt")
    if (!requirements.isFile) fail("Missing ${requirements.path}", 1)

    if (!venvPython.isFile) {
        System.err.println("Creating the project-local Python environment in .venv ...")
        run(listOf("python3", "-m", "venv", ".venv"), repoRoot, "python3 -m venv .venv")
    }

    val stamp = File(repoRoot, ".venv/.requirements-stamp")
    val expected = "${requirements.length()}:${requirements.lastModified()}"
    if (!stamp.isFile || stamp.readText().trim() != expected) {
        System.err.println("Installing worker requirements (first run downloads packages) ...")
        run(
            listOf(venvPython.path, "-m", "pip", "install", "--quiet", "--upgrade", "pip"),
            repoRoot,
            "pip upgrade",
        )
        run(
            listOf(venvPython.path, "-m", "pip", "install", "--quiet", "-r", requirements.path),
            repoRoot,
            "pip install -r worker/requirements.txt",
        )
        stamp.writeText(expected)
    }
}

val TEXT_SUFFIXES = listOf(".webarchive", ".md", ".markdown", ".mdown", ".mkd")

fun File.isSupported(): Boolean = TEXT_SUFFIXES.any { name.lowercase().endsWith(it) }

/** Expand one argument into the files it names: a glob, a directory, or a file. */
fun expand(argument: String): List<File> {
    val direct = File(argument)
    if (direct.isDirectory) {
        return direct.walkTopDown().filter { it.isFile && it.isSupported() }.sortedBy { it.path }.toList()
    }
    if (!argument.contains('*') && !argument.contains('?')) return listOf(direct)

    // The shell leaves a quoted glob unexpanded, so expand it here rather than
    // making the user remember which form they used.
    val pattern = File(argument)
    val parent = pattern.parentFile ?: File(".")
    val matcher = java.nio.file.FileSystems.getDefault().getPathMatcher("glob:${pattern.name}")
    return (parent.listFiles() ?: emptyArray())
        .filter { it.isFile && matcher.matches(File(it.name).toPath()) }
        .sortedBy { it.path }
}

fun percent(value: Double): String = String.format(Locale.ROOT, "%.1f%%", value * 100)

fun JsonObject.text(key: String): String? = this[key]?.jsonPrimitive?.contentOrNull
fun JsonObject.number(key: String): Double? = this[key]?.jsonPrimitive?.doubleOrNull
fun JsonObject.whole(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull
fun JsonObject.strings(key: String): List<String> =
    this[key]?.jsonArray?.mapNotNull { it.jsonPrimitive.contentOrNull } ?: emptyList()

fun renderReport(payload: String, inputs: List<File>, verbose: Boolean) {
    val report = try {
        Json.parseToJsonElement(payload).jsonObject
    } catch (error: Exception) {
        fail("The worker did not return a readable report: ${error.message}", 1)
    }

    // Without --verbose the report is prose: a reader who does not work with
    // these models gets the plain reading, not the percentage behind it.
    val lines = mutableListOf<String>()
    lines += if (inputs.size == 1) {
        "Read: ${inputs.single().path}"
    } else {
        "Read: ${inputs.size} files, scored together as one answer"
    }
    if (inputs.size > 1 && verbose) inputs.forEach { lines += "  ${it.path}" }
    lines += ""
    lines += "What the text detectors think"
    val detectors = report["text"]?.jsonArray.orEmpty()
    if (detectors.isEmpty()) {
        lines += "  (no text detector results)"
    }
    for (element in detectors) {
        val detector = element.jsonObject
        val name = detector.text("name") ?: "unknown"
        val score = detector.number("score")
        val label = detector.text("label") ?: ""
        val detail = detector.text("detail")
        if (score == null) {
            // A detector that could not run explains itself over as many lines
            // as its error needs, rather than being squeezed into parentheses.
            lines += "  $name: could not run"
            detail?.lines()?.filter { it.isNotBlank() }?.forEach { lines += "      ${it.trim()}" }
        } else if (verbose) {
            val head = "  $name: ${percent(score)} — $label"
            lines += if (detail.isNullOrBlank()) head else "$head ($detail)"
        } else {
            lines += "  $name: $label"
        }
    }

    val images = report["images"]?.jsonArray.orEmpty().map { it.jsonObject }
    if (images.isNotEmpty()) {
        lines += ""
        lines += "What the image detector thinks"
        for (image in images) {
            val index = image.whole("index") ?: 0
            val skipped = image.text("skipped_reason")
            val score = image.number("score")
            val label = image.text("label") ?: ""
            lines += when {
                skipped != null -> "  Image $index: skipped ($skipped)"
                score == null -> "  Image $index: unavailable"
                verbose -> "  Image $index: ${percent(score)} — $label"
                else -> "  Image $index: $label"
            }
        }

        val flagged = images.filter { it.strings("metadata_flags").isNotEmpty() }
        if (flagged.isNotEmpty()) {
            lines += ""
            lines += "What the image files say about themselves"
            for (image in flagged) {
                for (flag in image.strings("metadata_flags")) {
                    lines += "  Image ${image.whole("index") ?: 0}: $flag"
                }
            }
        }
    }

    val notes = report["warnings"]?.jsonArray.orEmpty().mapNotNull { it.jsonPrimitive.contentOrNull }
    if (notes.isNotEmpty()) {
        lines += ""
        lines += "Notes:"
        notes.forEach { lines += "  - $it" }
    }

    println(lines.joinToString(System.lineSeparator()))
}

val options = args.toList()
if (options.contains("--help") || options.contains("-h")) {
    println(USAGE)
    exitProcess(0)
}

val unknown = options.filter { it.startsWith("-") && it !in KNOWN_FLAGS }
if (unknown.isNotEmpty()) fail("Unknown option: ${unknown.first()}\n$USAGE", 2)

val positional = options.filterNot { it.startsWith("-") }
if (positional.isEmpty()) fail(USAGE, 2)

val inputs = positional.flatMap { expand(it) }.distinctBy { it.absolutePath }
if (inputs.isEmpty()) fail("No matching files: ${positional.joinToString(" ")}", 2)
val unsupported = inputs.filterNot { it.isSupported() }
if (unsupported.isNotEmpty()) {
    fail(
        "Unsupported file type: ${unsupported.first().path}\n" +
            "Expected one of: ${TEXT_SUFFIXES.joinToString(", ")}",
        2,
    )
}

if (!skipBootstrap) bootstrapPython()

val python = System.getenv("SLOP_DETECTOR_PYTHON") ?: venvPython.path
val workerDirectory = File(repoRoot, "worker")
val command = listOf(python, "-m", "slop_detector.main") +
    inputs.flatMap { listOf("--input", it.absolutePath) } +
    options.filter { it in KNOWN_FLAGS }

val builder = ProcessBuilder(command)
    .directory(if (workerDirectory.isDirectory) workerDirectory else repoRoot)
    .redirectError(ProcessBuilder.Redirect.INHERIT)
builder.environment()["HF_HOME"] = modelCache.path

val process = try {
    builder.start()
} catch (error: Exception) {
    fail("Could not start the Python worker ($python): ${error.message}", 1)
}
val payload = process.inputStream.bufferedReader().readText()
val code = process.waitFor()
if (code != 0) exitProcess(code)

renderReport(payload, inputs, options.contains("--verbose"))
